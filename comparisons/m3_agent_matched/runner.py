"""Matched-backbone M3-Agent adaptation; fresh perception, original graph operations."""
import argparse
import base64
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

ROOT = Path(os.environ.get("M3_EVAL_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUN = ROOT / "runs/m3_agent_matched_20261101"
EXP = ROOT / "experiments/m3_agent_matched_20261101"
sys.path.insert(0, str(RUN / "_source/oGMemory"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image
from reference import CachedEmbeddings, load_reference, normalise_memory
from core.models import RequestContext
from core.react_loop import ReactLoop, ReactTaskContext
from extraction.operators.builtin.episodic_memory_generator import _audio_part_from_node, _image_part_from_node
from m3.config import load_m3_runtime_config
from m3.episodic_memory import _index_asr_segments, _index_audio_records, _index_frame_records, _retryable_generation_error
from m3.index_episodic import build_embedder
from m3.llm_factory import build_episodic_llm, build_qa_llm
from m3.load_clips import build_graph_service
from m3.qa import load_questions, _has_protocol_scaffolding, _salvage_scaffolding
from core.llm_output import final_text
from m3.temporal import resolve_question_scope, temporal_prompt

logger = logging.getLogger("m3_agent_matched")
SCHEMA = {
    "type": "object", "properties": {
        "video_descriptions": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "high_level_conclusions": {"type": "array", "items": {"type": "string"}},
    }, "required": ["video_descriptions", "high_level_conclusions"],
}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temporary.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def anchor_update(node_id, kind, info):
    # Upstream add_*_node retains the contents list by reference. Journal the
    # current observation, not a list that future graph updates can extend.
    return {"id": node_id, "type": kind, "info": copy.deepcopy(info)}


def crop_image(frame, bbox):
    part = _image_part_from_node(frame, "auto")
    if not part or not part["image_url"]["url"].startswith("data:"):
        raise ValueError("Matched perception requires local image bytes")
    image = Image.open(BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
    x1, y1, x2, y2 = bbox
    # Shared CV backend contract: coordinates are normalized, not pixel indices.
    x1, x2 = x1 * image.width, x2 * image.width
    y1, y2 = y1 * image.height, y2 * image.height
    bounds = (max(0, math.floor(x1)), max(0, math.floor(y1)),
              min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Invalid face bounding box")
    output = BytesIO()
    image.crop(bounds).save(output, format="JPEG", quality=80)
    return base64.b64encode(output.getvalue()).decode()


def prepare_clips(video, store, ctx, ref):
    """Recompute identity matching from this run's newly generated perception."""
    work = RUN / video
    frame_index = _index_frame_records(work / "frames.jsonl")
    audio_index = _index_audio_records(work / "audio.jsonl")
    transcripts = _index_asr_segments(store, ctx, audio_index)
    face_rows = {}
    for row in rows(work / "faces.jsonl"):
        face_rows[row["frame_node_id"].replace("-", "")] = row["child_node_ids"]
    voices = {}
    for row in rows(work / "speaker_embeddings.jsonl"):
        node = store.get_extracted_node(row["speaker_node_id"], ctx)
        voices.setdefault(int(row["clip_id"]), []).append(node.extracted_content)
    anchors = ref.VideoGraph(**ref.memory_config)
    clips, updates = [], []
    manifest = rows(work / "data.jsonl")[0]
    for record in sorted(manifest["raw_nodes"], key=lambda r: int(r["clip_id"])):
        clip = int(record["clip_id"])
        key = f"{video}|{clip}"
        raw = store.get_raw_node(record["raw_node_id"], ctx)
        frames = [store.get_raw_node(n, ctx) for n in frame_index[key]]
        tagged_faces, tagged_voices, current_updates = {}, {}, []
        for frame in frames:
            for face_id in face_rows.get(str(frame.node_id).replace("-", ""), []):
                face = store.get_extracted_node(face_id, ctx).extracted_content
                if not face.get("embedding"):
                    continue
                cropped = crop_image(frame, face["bbox"])
                info = {"embeddings": [face["embedding"]], "contents": [cropped]}
                matches = anchors.search_img_nodes(info)
                if matches:
                    node_id = matches[0][0]
                    anchors.update_node(node_id, info)
                else:
                    node_id = anchors.add_img_node(info)
                current_updates.append(anchor_update(node_id, "img", info))
                candidate = {"confidence": float(face.get("confidence", 0)), "crop": cropped}
                if candidate["confidence"] > tagged_faces.get(node_id, {}).get("confidence", -1):
                    tagged_faces[node_id] = candidate
        speaker_ids = {}
        for voice in voices.get(clip, []):
            info = {"embeddings": [voice["speaker_embedding"]], "contents": [voice["speaker_id"]]}
            matches = anchors.search_voice_nodes(info)
            if matches:
                node_id = matches[0][0]
                anchors.update_node(node_id, info)
            else:
                node_id = anchors.add_voice_node(info)
            speaker_ids[voice["speaker_id"]] = node_id
            current_updates.append(anchor_update(node_id, "voice", info))
        for segment in transcripts.get(key, []):
            speaker = speaker_ids.get(segment.get("speaker_id"))
            tag = f"<voice_{speaker}>" if speaker is not None else "unattributed"
            tagged_voices.setdefault(tag, []).append({
                "start_time": segment["start_sec"], "end_time": segment["end_sec"],
                "asr": segment["text"],
            })
        parts = [{"type": "text", "text": ref.prompt + f"\nvideo_id: {video}\nclip_id: {clip}\n"
                  + "Voice features (clip-relative seconds):\n" + json.dumps(tagged_voices, ensure_ascii=False)}]
        parts.extend(_image_part_from_node(frame, "auto") for frame in frames)
        selected_faces = sorted(tagged_faces.items(), key=lambda x: (-x[1]["confidence"], x[0]))[:8]
        for node_id, data in selected_faces:
            parts += [{"type": "text", "text": f"<face_{node_id}>:"},
                      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data["crop"], "detail": "auto"}}]
        audio = store.get_raw_node(audio_index[key], ctx)
        audio_part = _audio_part_from_node(audio)
        if audio_part:
            parts.append(audio_part)
        messages = [{"role": "user", "content": parts}]
        clips.append({
            "clip_id": clip, "messages": messages, "messages_sha256": digest(messages),
            "start_sec": raw.metadata["start_sec"],
            "duration_sec": raw.metadata["duration_sec"],
            "face_crops": len(selected_faces), "available_face_ids": sorted(tagged_faces),
            "audio_attached": bool(audio_part),
        })
        updates.append({"clip_id": clip, "updates": current_updates})
    return clips, updates, anchors.next_node_id


def build_prefix(ref, updates, memories, cutoff, anchor_count):
    """Replay only admissible facts; future semantic reinforcement cannot leak in."""
    graph = ref.VideoGraph(**ref.memory_config)
    graph.next_node_id = anchor_count
    for record in updates:
        if record["clip_id"] > cutoff:
            continue
        for item in record["updates"]:
            node_id = item["id"]
            if node_id not in graph.nodes:
                node = graph.Node(node_id, item["type"])
                graph.nodes[node_id] = node
                node.metadata["contents"] = []
            graph.update_node(node_id, copy.deepcopy(item["info"]))
    for clip, memory in sorted(memories.items()):
        if clip <= cutoff:
            ref.process_memories(graph, memory["video_descriptions"], clip, type="episodic")
            ref.process_memories(graph, memory["high_level_conclusions"], clip, type="semantic")
    graph.refresh_equivalences()
    return graph


@dataclass
class Task(ReactTaskContext):
    question: str
    temporal_scope: dict = field(default_factory=dict)
    search_history: list = field(default_factory=list)


class Controller(ReactLoop):
    def __init__(self, llm, ref, graph, cutoff, coverage):
        super().__init__(llm=llm, max_iterations=6, timeout_seconds=180)
        self.ref, self.graph, self.cutoff = ref, graph, cutoff
        self.coverage = coverage
        self.answer = ""
        self.current_clips = []

    def _get_tools(self):
        return [
            {"name": "Search", "description": "Search identity-linked episodic and semantic memory.",
             "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "Answer", "description": "Return a concise answer grounded in retrieved memory.",
             "input_schema": {"type": "object", "properties": {"final_answer": {"type": "string"}}, "required": ["final_answer"]}},
        ]

    def _build_messages(self, ctx, task_ctx):
        self.task = task_ctx
        system = (
            "You are given a question and relevant knowledge. Use the native Search tool if "
            "knowledge is insufficient, or Answer when you can answer. Do not print Action "
            "labels instead of calling tools. Search queries should differ from earlier ones. "
            "You can find a name/ID mapping by asking 'What is the character id of NAME' or "
            "'What is the name of <character_0>'. After obtaining a mapping, prefer character "
            "IDs for searching. Only names, not character IDs, should appear in the final answer. "
            "Ground answers in evidence; do not invent missing facts. "
            + temporal_prompt(task_ctx.temporal_scope)
        )
        if self.coverage["failed_clips"]:
            system += f"\nSome clip memories are unavailable: {self.coverage['failed_clips']}. Missing evidence is not evidence of absence."
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"Question: {task_ctx.question}\nSearched knowledge: {{}}"}]

    def _execute_custom_tool(self, name, input, ctx):
        if name == "Answer":
            answer = final_text(str(input.get("final_answer", "")))
            if not answer or _has_protocol_scaffolding(answer):
                return {"error": "A nonempty answer without protocol scaffolding is required"}
            self.answer = answer
            return {"status": "answered"}
        if name != "Search":
            return {"error": f"Unknown tool: {name}"}
        query = input.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"error": "Search requires query"}
        memory_wise = "character id" in query
        memories, current, scores = self.ref.search(
            self.graph, query, self.current_clips, topk=20 if memory_wise else 2,
            threshold=0 if memory_wise else 0.5, mem_wise=memory_wise, before_clip=self.cutoff)
        self.current_clips = current
        record = {"query": query, "memories": memories,
                  "clip_scores": {str(k): float(v) for k, v in scores.items()},
                  "temporal_scope": self.task.temporal_scope}
        self.task.search_history.append(record)
        return {"memories": memories, "empty": not bool(memories)}

    def _parse_output(self, content):
        value = final_text(content)
        return [value] if value else None

    def _output_after_tools(self):
        return [self.answer] if self.answer else None

    def _allow_tool_iteration_extension(self):
        return False

    def _build_final_instruction(self):
        return "This is the final iteration. Answer from existing evidence, or explicitly state insufficiency; do not guess unsupported facts."


def run_video(video, config, limit_clips=None, skip_qa=False):
    random.seed(42)
    work = RUN / video
    if (work / "m3_qa_results.jsonl").exists():
        raise RuntimeError("QA output exists; refusing to rewrite completed input journals or answers")
    receipt = work / "perception_exit_code.txt"
    if not receipt.exists() or receipt.read_text().strip() != "0":
        raise RuntimeError("Fresh perception must finish successfully before this stage")
    runtime = load_m3_runtime_config(str(config))
    store, _ = build_graph_service("sql", runtime.graph_connection_string)
    account = "m3-agent-matched-20261101"
    ctx = RequestContext(account_id=account, user_id="m3bench-user", agent_id="m3bench-agent",
                         session_id=video, trace_id="m3-matched")
    embeddings = CachedEmbeddings(build_embedder(None, config_path=str(config)))
    ref = load_reference(RUN / "_source/m3-agent", embeddings)
    save(EXP / "reference_definitions.json", ref.receipt)
    clips, updates, anchor_count = prepare_clips(video, store, ctx, ref)
    save(work / "identity_updates.json", updates)
    save(work / "clip_inputs.json", [{k: v for k, v in c.items() if k != "messages"} for c in clips])
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    llm = build_episodic_llm(config_path=str(config))
    generation_dir = work / "m3_memories"
    generation_dir.mkdir(exist_ok=True)

    def generate(clip):
        out = generation_dir / f"{clip['clip_id']}.json"
        if out.exists():
            old = json.loads(out.read_text())
            if old["messages_sha256"] != clip["messages_sha256"] or old["config_sha256"] != config_hash:
                raise ValueError("Refusing to mix changed model/media inputs in a resumed memory")
            normalise_memory(old["memory"])
            return old
        failed_receipt = generation_dir / f"{clip['clip_id']}.failure.json"
        if failed_receipt.exists():
            old = json.loads(failed_receipt.read_text())
            if old.get("messages_sha256") != clip["messages_sha256"] or old.get("config_sha256") != config_hash:
                raise ValueError("Failed receipt has a different input/configuration")
            return old
        history = []
        invalid_retries = transient_retries = 0
        while True:
            try:
                value = normalise_memory(llm.complete_multimodal(clip["messages"], SCHEMA))
                record = {"video_id": video, "clip_id": clip["clip_id"], "memory": value,
                          "messages_sha256": clip["messages_sha256"], "config_sha256": config_hash,
                          "retry_errors": history, "status": "completed"}
                save(out, record)
                logger.info("Memory complete: %s clip %s", video, clip["clip_id"])
                return record
            except Exception as error:
                history.append({"type": type(error).__name__, "error": str(error)})
                if isinstance(error, ValueError) and invalid_retries < 1:
                    invalid_retries += 1
                    continue
                if _retryable_generation_error(error) and transient_retries < 3:
                    time.sleep(min(60, 10 * 2 ** transient_retries))
                    transient_retries += 1
                    continue
                break
        record = {"video_id": video, "clip_id": clip["clip_id"], "status": "failed", "errors": history,
                  "messages_sha256": clip["messages_sha256"], "config_sha256": config_hash}
        save(failed_receipt, record)
        return record

    selected = clips if limit_clips is None else clips[:limit_clips]
    # Validate ALL receipts before submitting any paid model work.
    for clip in selected:
        for path in (generation_dir / f"{clip['clip_id']}.json",
                     generation_dir / f"{clip['clip_id']}.failure.json"):
            if path.exists():
                old = json.loads(path.read_text())
                if old.get("messages_sha256") != clip["messages_sha256"] or old.get("config_sha256") != config_hash:
                    raise ValueError("Cached memory input/configuration mismatch before scheduling")
    with ThreadPoolExecutor(max_workers=4) as pool:
        generated = list(pool.map(generate, selected))
    memories = {r["clip_id"]: r["memory"] for r in generated if r["status"] == "completed"}
    failed = [c["clip_id"] for c in clips if c["clip_id"] not in memories]
    coverage = {"expected_clips": len(clips), "completed_clips": len(memories), "failed_clips": failed}
    save(work / "memory_summary.json", coverage)
    if limit_clips is not None or skip_qa:
        return coverage
    if len(memories) / len(clips) < 0.95:
        raise RuntimeError("Memory coverage below 95%; refusing low-coverage QA")
    random.seed(42)
    full = build_prefix(ref, updates, memories, max(c["clip_id"] for c in clips), anchor_count)
    save(work / "m3_graph.json", {
        "nodes": [{"id": n.id, "type": n.type, "embeddings": n.embeddings, "metadata": n.metadata}
                  for n in full.nodes.values()],
        "edges": [[a, b, weight] for (a, b), weight in full.edges.items()],
        "character_mappings": full.character_mappings,
    })
    save(work / "embedding_cache.json", embeddings.cache)
    qa_llm = build_qa_llm(config_path=str(config))
    qa_output = work / "m3_qa_results.jsonl"
    if qa_output.exists():
        raise RuntimeError("QA output already exists; refusing to replace first answers")
    questions = [q for q in load_questions(Path("/data/m3_data/annotations/robot.json")) if q["video_id"] == video]

    def lookup(second):
        candidates = [c for c in clips if c["start_sec"] <= second < c["start_sec"] + c["duration_sec"]]
        if not candidates:
            return None
        c = max(candidates, key=lambda c: c["start_sec"])
        return {"clip_id": c["clip_id"], "source_start_sec": c["start_sec"],
                "source_end_sec": c["start_sec"] + c["duration_sec"], "source_duration_sec": c["duration_sec"]}

    with qa_output.open("x") as f:
        for q in questions:
            cutoff, seconds, scope = resolve_question_scope(
                q["before_clip"], q["timestamp"], policy="current_clip", clip_lookup=lookup)
            # Source interval fallback must never open access to future clips.
            admissible = [c["clip_id"] for c in clips
                          if (seconds is None or c["start_sec"] + c["duration_sec"] <= seconds + 1e-6)
                          and (cutoff is None or c["clip_id"] <= cutoff)]
            bound = max(admissible, default=-1)
            random.seed(42 + bound)
            graph = build_prefix(ref, updates, memories, bound, anchor_count)
            task = Task(q["question"], scope)
            adapter = Controller(qa_llm, ref, graph, bound, coverage)
            error, prediction, finish, iterations = "", "", "", 0
            try:
                result = adapter.run(ctx, task)
                finish, iterations = result.finish_reason, result.iterations
                prediction = str(result.output[0]).strip() if result.output else ""
                if not result.completed or not prediction:
                    error = f"loop did not complete ({finish})"
                    prediction = ""
                elif _has_protocol_scaffolding(prediction):
                    prediction = _salvage_scaffolding(prediction)
                    if len(prediction) < 3:
                        prediction, error = "", "predicted_answer contains protocol scaffolding"
            except Exception as exc:
                error, finish = str(exc), "error"
            row = {k: q[k] for k in ("video_id", "question_id", "question_index", "question", "type", "timestamp", "before_clip")}
            row.update(expected_answer=q["answer"], predicted_answer=prediction, error=error,
                       finish_reason=finish, iterations=iterations, search_rounds=len(task.search_history),
                       search_history=task.search_history, temporal_scope=scope, memory_coverage=coverage)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            logger.info("QA complete: %s %s", q["question_id"], finish)
    save(work / "adapted_summary.json", {
        "video_id": video, "memory_coverage": coverage,
        "qa_records": len(questions), "valid_answers": sum(not r["error"] for r in rows(qa_output)),
        "node_types": {kind: sum(n.type == kind for n in full.nodes.values())
                       for kind in ["img", "voice", "episodic", "semantic"]},
        "embedding_logical_calls": embeddings.calls, "embedding_texts": embeddings.texts,
    })
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--limit-clips", type=int)
    parser.add_argument("--skip-qa", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_video(args.video, EXP / "private/config.yaml", args.limit_clips, args.skip_qa)
