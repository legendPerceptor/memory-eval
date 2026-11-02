"""QA-only reference text-action controller on the same newly built M3 memories.

Uses the upstream system/instruction/pattern/consumer definitions verbatim.
The common model, six-round and 180-second budgets are explicit adaptations.
This arm does not regenerate perception/memory or replace native-tool answers.
"""
import ast
import copy
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["M3_EVAL_ROOT"])
RUN = ROOT / "runs" / os.environ.get("M3_MATCHED_NAME", "m3_agent_matched_20261101")
EXP = ROOT / "experiments" / os.environ.get("M3_REFERENCE_NAME", "m3_agent_reference_text_20261101")
EXP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "_adapter"))
from runner import build_prefix, rows, save
from reference import CachedEmbeddings, load_reference
from m3.llm_factory import build_qa_llm
from m3.index_episodic import build_embedder
from m3.qa import load_questions, _has_protocol_scaffolding
from m3.temporal import resolve_question_scope
from core.llm_output import final_text

logger = logging.getLogger("m3.reference_text")
CONFIG = ROOT / "experiments" / os.environ.get("M3_MATCHED_NAME", "m3_agent_matched_20261101") / "private/config.yaml"
OUTPUT = Path(os.environ.get("M3_REFERENCE_OUTPUT", str(ROOT / "runs/m3_agent_reference_text_20261101")))
LAST_INSTRUCTION = "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"


def controller_definitions(ref):
    path = RUN / "_source/m3-agent/m3_agent/control.py"
    tree = ast.parse(path.read_text())
    names = {"system_prompt", "instruction", "pattern"}
    selected = [node for node in tree.body if (
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
    ) or isinstance(node, ast.FunctionDef) and node.name == "consumer"]
    namespace = {"re": re, "json": json, "search": ref.search, "processing_config": {"topk": 2}}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    assert names.issubset(namespace) and "consumer" in namespace
    return namespace


def run_video(video):
    source = RUN / video
    work = OUTPUT / video
    work.mkdir(parents=True, exist_ok=True)
    output = work / "m3_qa_results.jsonl"
    if output.exists():
        raise RuntimeError("Refusing to overwrite text-controller predictions")
    deadline = time.monotonic() + 7200
    while not (source / "embedding_cache.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("M3 memories not ready")
        if (source / "memory_summary.json").exists():
            state = json.loads((source / "memory_summary.json").read_text())
            if state["completed_clips"] / state["expected_clips"] < .95:
                raise RuntimeError("M3 memory coverage below threshold")
        time.sleep(15)
    coverage = json.loads((source / "memory_summary.json").read_text())
    clips = json.loads((source / "clip_inputs.json").read_text())
    updates = json.loads((source / "identity_updates.json").read_text())
    memories = {int(p.stem): json.loads(p.read_text())["memory"]
                for p in (source / "m3_memories").glob("*.json") if ".failure." not in p.name}
    embeddings = CachedEmbeddings(build_embedder(None, config_path=str(CONFIG)))
    embeddings.cache = json.loads((source / "embedding_cache.json").read_text())
    ref = load_reference(RUN / "_source/m3-agent", embeddings)
    control = controller_definitions(ref)
    anchors = 1 + max((u["id"] for c in updates for u in c["updates"]), default=-1)
    llm = build_qa_llm(config_path=str(CONFIG))
    questions = [q for q in load_questions(Path("/data/m3_data/annotations/robot.json"))
                 if q["video_id"] == video]

    def lookup(second):
        found = [c for c in clips if c["start_sec"] <= second < c["start_sec"] + c["duration_sec"]]
        if not found:
            return None
        c = max(found, key=lambda c: c["start_sec"])
        return {"clip_id": c["clip_id"], "source_start_sec": c["start_sec"],
                "source_end_sec": c["start_sec"] + c["duration_sec"], "source_duration_sec": c["duration_sec"]}

    with output.open("x") as stream:
        for q in questions:
            cutoff, seconds, scope = resolve_question_scope(q["before_clip"], q["timestamp"],
                                                           policy="current_clip", clip_lookup=lookup)
            admissible = [c["clip_id"] for c in clips if
                          (cutoff is None or c["clip_id"] <= cutoff)
                          and (seconds is None or c["start_sec"] + c["duration_sec"] <= seconds + 1e-6)]
            bound = max(admissible, default=-1)
            random.seed(42 + bound)
            graph = build_prefix(ref, updates, memories, bound, anchors)
            control["load_video_graph"] = lambda path: copy.deepcopy(graph)
            data = {"finish": False, "before_clip": bound, "mem_path": "in-memory-prefix",
                    "currenr_clips": [], "conversations": [
                        {"role": "system", "content": control["system_prompt"].format(question=q["question"])},
                        {"role": "user", "content": "Searched knowledge: {}"}]}
            error, finish, prediction = "", "max_iterations", ""
            began = time.monotonic()
            history = []
            for iteration in range(1, 7):
                if iteration > 1 and time.monotonic() - began > 180:
                    finish = "timeout"
                    break
                data["conversations"][-1]["content"] += control["instruction"]
                if iteration == 6:
                    data["conversations"][-1]["content"] += LAST_INSTRUCTION
                try:
                    calls, content = llm.complete_with_tools_messages(
                        data["conversations"], tools=None, tool_choice="none")
                    if calls:
                        raise ValueError("Text protocol unexpectedly returned native tools")
                    data["conversations"].append({"role": "assistant", "content": content})
                    control["consumer"](data)
                    history.append({"iteration": iteration, "model_text": content,
                                    "observation": data["conversations"][-1]["content"] if not data["finish"] else None})
                    if data["finish"]:
                        prediction = final_text(data.get("response", ""))
                        finish = "done"
                        break
                except Exception as exc:
                    error, finish = str(exc), "error"
                    break
            if finish != "done" or not prediction or _has_protocol_scaffolding(prediction):
                error = error or f"loop did not complete ({finish})"
                prediction = ""
            row = {k: q[k] for k in ("video_id", "question_id", "question_index", "question",
                                     "type", "timestamp", "before_clip")}
            row.update(expected_answer=q["answer"], predicted_answer=prediction, error=error,
                       finish_reason=finish, iterations=iteration, temporal_scope=scope,
                       memory_coverage=coverage, search_history=history,
                       search_rounds=sum(h["observation"] is not None for h in history))
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            logger.info("%s %s", q["question_id"], finish)
    return {"video": video, "qa_records": len(questions),
            "valid_answers": sum(not q["error"] for q in rows(output))}


def main():
    manifest = json.loads((RUN / "sample_manifest.json").read_text())
    videos = manifest["selected_video_ids"]
    OUTPUT.mkdir(exist_ok=True)
    receipts = []
    for video in videos:
        save(EXP / "status.json", {"phase": "text_controller_qa", "video": video, "completed": receipts})
        receipts.append(run_video(video))
    merged = OUTPUT / "all_qa.jsonl"
    merged.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n"
                             for v in videos for q in rows(OUTPUT / v / "m3_qa_results.jsonl")))
    save(EXP / "status.json", {"phase": "grading", "completed": receipts})
    with (EXP / "judge.log").open("w") as log:
        result = subprocess.run(
            [str(ROOT / "oGMemory/.venv/bin/python"), "-m", "m3.eval", "--config", str(CONFIG),
             "--annotations", "/data/m3_data/annotations/robot.json", "--video", ",".join(videos),
             "--input-qa-results", str(merged), "--output-dir", str(OUTPUT / "_aggregate")],
            cwd=RUN / "_source/oGMemory", stdout=log, stderr=subprocess.STDOUT)
    summary = json.loads((OUTPUT / "_aggregate/m3_eval_summary.json").read_text())
    save(EXP / "summary.json", summary)
    save(EXP / "status.json", {"phase": "finished", "completed": receipts,
                             "mean_score": summary["overall_mean_score"], "judge_exit_code": result.returncode})
    (EXP / "exit_code.txt").write_text(str(result.returncode) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        main()
    except Exception as error:
        save(EXP / "status.json", {"phase": "needs_attention", "error": str(error)})
        raise
