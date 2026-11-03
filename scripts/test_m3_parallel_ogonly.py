"""Offline safety/measurement checks for the oGMemory-only coordinator."""

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

PATH = Path(__file__).with_name("run_m3_parallel_ogonly.py")
spec = importlib.util.spec_from_file_location("ogonly_test_driver", PATH)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_environment_is_parallel_and_not_global_mutation(monkeypatch):
    monkeypatch.delenv("OGMEM_M3_MAX_WORKERS", raising=False)
    env = driver.environment("og_qa", "study_09")
    assert env["OGMEM_M3_MAX_WORKERS"] == "4"
    assert env["OGMEM_M3_FACE_WORKERS"] == "2"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    import os

    assert "OGMEM_M3_MAX_WORKERS" not in os.environ


def test_resource_usage_keeps_failed_calls_and_unknowns(tmp_path):
    rows = [
        {
            "requested_model": "MiniMax-M3.1-Flash",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "status": "returned",
        },
        {"requested_model": "MiniMax-M3.1-Flash", "usage": None, "status": "error"},
        {"requested_model": "other", "usage": {"prompt_tokens": 1000}},
    ]
    (tmp_path / "usage-og_qa-v.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (tmp_path / "stages").mkdir()
    (tmp_path / "stages/v-og_qa.json").write_text(
        json.dumps({"seconds": 50, "exit_code": 2})
    )
    result = driver.resources(tmp_path, ["v"], ["og_qa"])
    assert result == {
        "calls": 2,
        "unknown_usage_calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "wall_seconds_including_quota_wait": 50,
        "failed_stages": 1,
    }


def test_no_reference_inference_phases():
    import ast

    tree = ast.parse(PATH.read_text())
    worker = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"
    )
    assert "reference_text" not in ast.unparse(worker)
    assert "import runner" not in ast.unparse(worker)
    run = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    assert '("pipeline", "index", "og_qa")' in ast.unparse(
        run
    ) or "('pipeline', 'index', 'og_qa')" in ast.unparse(run)


def test_concurrent_usage_attribution(tmp_path):
    root = PATH.parent.parent
    code = textwrap.dedent("""
        import importlib.util, json, os, time
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from providers.llm.openai_llm import OpenAILLM
        from m3._parallel import call_scope
        def fake(self, **kwargs):
            time.sleep(.005)
            return SimpleNamespace(id='fake', model='test', usage=None, choices=[])
        OpenAILLM._create_completion=fake
        spec=importlib.util.spec_from_file_location('audit', os.environ['AUDIT_FILE'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        client=SimpleNamespace(_base_url='http://localhost',_client=SimpleNamespace(timeout=10))
        def unit(i):
            with call_scope(qa_video_id=f'video-{i}',qa_question_index=i):
                OpenAILLM._create_completion(client,model='test',messages=[])
        with ThreadPoolExecutor(4) as pool:list(pool.map(unit,range(20)))
        rows=[json.loads(s) for s in open(os.environ['M3_USAGE_LOG'])]
        assert len(rows)==20
        assert all(r['qa_video_id']==f"video-{r['qa_question_index']}" for r in rows)
    """)
    import os

    env = {
        **os.environ,
        "PYTHONPATH": os.environ.get("M3_TEST_SOURCE", str(root / "oGMemory")),
        "AUDIT_FILE": str(root / "scripts/instrumentation/sitecustomize.py"),
        "M3_USAGE_LOG": str(tmp_path / "usage.jsonl"),
    }
    subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True
    )
