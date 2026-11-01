#!/usr/bin/env python3
"""Isolated SQL fault-injection test; never a benchmark score or live LLM run."""
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import psycopg2
import yaml
from psycopg2 import sql
from psycopg2.extensions import parse_dsn, make_dsn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oGMemory"))
from m3 import episodic_memory
from m3.run_video_pipeline import main as pipeline_main

EXP = ROOT / "experiments/m3_study18_recovery_20261101"
WORK = ROOT / "runs/m3_partial_fault_injection_20261101/study_18"
SOURCE = ROOT / "runs/m3_flash_medium_20261031/study_18"
DB = "ogmem_m3_partial_fault_20261101"


def main():
    config = yaml.safe_load((ROOT / "experiments/m3_flash_medium_20261031/private/config.yaml").read_text())
    params = parse_dsn(config["storage"]["connection_string"])
    admin = psycopg2.connect(**{**params, "dbname": "postgres"})
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute("select 1 from pg_database where datname=%s", (DB,))
        assert not cursor.fetchone(), "Refuse to overwrite an existing fault-injection database"
        cursor.execute(sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
            sql.Identifier(DB), sql.Identifier(params["dbname"])))
    admin.close()
    for key in ("storage", "vector_db"):
        config[key]["connection_string"] = make_dsn(**{**params, "dbname": DB})
    for role in ("episodic_llm", "qa_llm", "judge_llm"):
        config["m3"][role]["api_key"] = "fault-test-no-live-llm"
    config["llm"]["api_key"] = "fault-test-no-live-llm"
    cfg = EXP / "private/fault-test.yaml"
    cfg.write_text(yaml.safe_dump(config))
    cfg.chmod(0o600)
    WORK.mkdir(parents=True)
    for path in SOURCE.iterdir():
        if path.is_file() and path.name != "pipeline.log":
            shutil.copy2(path, WORK / path.name)
    annotations = json.loads(Path("/data/m3_data/annotations/robot.json").read_text())
    one = {"study_18": {**annotations["study_18"], "qa_list": [annotations["study_18"]["qa_list"][-1]]}}
    question_path = WORK / "test_question.json"
    question_path.write_text(json.dumps(one))
    attempts = []
    original = episodic_memory.generate_episodic_for_clip

    def injected_failure(**kwargs):
        if kwargs["video_raw_node"].metadata["clip_id"] == 52:
            attempts.append(52)
            raise RuntimeError("Invalid episodic memory: injected persistent invalid output")
        return original(**kwargs)

    class ForbiddenLLM:
        def complete_multimodal(self, *args, **kwargs):
            raise AssertionError("Fault test must never call a live memory model")

    class QAStub:
        def __init__(self):
            self.calls = 0
        def complete_with_tools_messages(self, messages, **kwargs):
            self.calls += 1
            assert "83 of 84" in messages[0]["content"]
            assert "52" in messages[0]["content"]
            if self.calls == 1:
                return [{"id": "test-search", "tool": "Search",
                         "input": {"query": "objects on the desk", "top_k": 8}}], ""
            return [{"id": "test-answer", "tool": "Answer",
                     "input": {"final_answer": "Fault-injection test answer; not a benchmark prediction."}}], ""

    disabled = ["trim", "annotate", "preprocess", "load_clips", "extract_audio",
                "extract_frames", "detect_faces", "voice_asr", "speaker_diarization",
                "speaker_embedding", "face_clustering", "voice_segmentation", "voice_clustering"]
    argv = ["--video", "/data/m3_data/videos/robot/study_18.mp4", "--video-id", "study_18",
            "--seconds", "0", "--work-dir", str(WORK), "--config", str(cfg),
            "--pipeline", str(ROOT / "oGMemory/config/m3/pipelines/video_episodic_full.yaml"),
            "--graph-store", "sql", "--account-id", "m3-flash-medium-20261031", "--no-smart-resume",
            "--set", "episodic_memory.params.retry_failed=true",
            "--set", "episodic_memory.params.workers=4",
            "--set", f"qa.params.annotations={question_path}", "--set", "qa.params.video=study_18"]
    for stage in disabled:
        argv += ["--disable-stage", stage]
    with patch.object(episodic_memory, "generate_episodic_for_clip", injected_failure), \
         patch.object(episodic_memory, "_build_llm", lambda args: ForbiddenLLM()), \
         patch("m3.qa.build_llm", lambda *args, **kwargs: QAStub()):
        rc = pipeline_main(argv)
    summary = json.loads((WORK / "pipeline_summary.json").read_text())
    answers = [json.loads(s) for s in (WORK / "m3_qa_results.jsonl").read_text().splitlines()]
    assert attempts == [52, 52], attempts
    assert rc == 2
    assert summary["stages"]["episodic_memory"]["status"] == "degraded"
    assert summary["stages"]["build_index"]["ok"] and summary["stages"]["qa"]["ok"]
    assert len(answers) == 1 and not answers[0]["error"]
    assert answers[0]["memory_coverage"]["partial"]
    assert answers[0]["memory_coverage"]["missing_clip_ids"] == [52]
    with psycopg2.connect(config["vector_db"]["connection_string"]) as db:
        with db.cursor() as cursor:
            cursor.execute("select count(*) from m3_episodic_vectors where video_id='study_18'")
            count = cursor.fetchone()[0]
    assert count == 83, count
    result = {"status": "passed", "test_kind": "SQL integration with injected persistent clip failure and stub answer",
              "live_memory_llm_calls": 0, "live_qa_llm_calls": 0,
              "local_embedding_service_used": True, "injected_attempts": attempts,
              "usable_memories_and_vectors": 83, "expected_clips": 84,
              "downstream_index_and_qa_executed": True, "qa_coverage_disclosed": True,
              "exit_code": rc, "original_database_unchanged": True, "test_database": DB}
    (EXP / "partial-fault-injection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
