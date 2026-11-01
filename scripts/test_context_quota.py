import json
import sys
from types import SimpleNamespace

import pytest

import run_context_experiment as runner
sys.path.insert(0, str(runner.ROOT / "oGMemory"))

QUOTA = "Error code: 429 已达到 Token Plan 用量上限 (2056)"


def test_quota_detection_does_not_treat_every_429_as_exhausted():
    assert runner.quota_exhausted(RuntimeError(QUOTA))
    assert runner.quota_exhausted("Token Plan quota exhausted (2056)")
    assert not runner.quota_exhausted("429 Too many requests; retry later")
    assert not runner.quota_exhausted("2056 matches found")
    assert not runner.quota_exhausted("context budget exceeded")


@pytest.mark.parametrize("last_question", [False, True])
def test_qa_stops_after_first_quota_error_without_fabricating_remaining_rows(
    tmp_path, monkeypatch, last_question,
):
    from m3 import qa, config, control_adapter

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "sentinel"))
    for key in ("OGMEM_GRAPH_CONNECTION_STRING", "M3_AUDIT_VIDEO", "M3_AUDIT_QUESTION_INDEX"):
        monkeypatch.setenv(key, "test")
    monkeypatch.setattr(config, "load_m3_runtime_config",
                        lambda _: SimpleNamespace(graph_connection_string="unit-test-only"))
    monkeypatch.setattr(runner, "DEADLINE", float("inf"))
    run = tmp_path / "runs/test"
    run.mkdir(parents=True)
    calls = []

    def unavailable(self, ctx, task):
        calls.append(1)
        raise RuntimeError(QUOTA)

    monkeypatch.setattr(control_adapter.OGMemoryControlAdapter, "run", unavailable)

    def fake_qa(argv):
        adapter = object.__new__(control_adapter.OGMemoryControlAdapter)
        task = SimpleNamespace(search_stats={})
        with pytest.raises(RuntimeError, match="2056"):
            adapter.run(None, task)
        path = run / "study_05/m3_qa_results.jsonl"
        path.write_text(json.dumps({"error": QUOTA}) + "\n")
        if not last_question:
            adapter.run(None, task)  # SystemExit(75), caught by worker
            raise AssertionError("Must not start a second model call")
        return 2

    monkeypatch.setattr(qa, "main_react_qa", fake_qa)
    assert runner.worker(SimpleNamespace(name="test", video="study_05")) == 75
    assert calls == [1]
    assert len((run / "study_05/m3_qa_results.jsonl").read_text().splitlines()) == 1
    assert json.loads((run / "study_05/quota_stop.json").read_text())["provider_code"] == 2056


def test_judge_persists_first_quota_failure_then_stops(tmp_path, monkeypatch):
    from m3 import eval as evaluation

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "sentinel"))
    monkeypatch.setattr(runner, "DEADLINE", float("inf"))
    calls = []
    directory = tmp_path / "runs/test/study_05/judge"
    directory.mkdir(parents=True)

    def unavailable(**kwargs):
        calls.append(1)
        return 0., QUOTA, True

    monkeypatch.setattr(evaluation, "judge_answer", unavailable)

    def fake_eval(argv):
        result = evaluation.judge_answer()
        (directory / "m3_eval_results.jsonl").write_text(json.dumps({"judge_error": result[2]}) + "\n")
        evaluation.judge_answer()  # must stop before the second upstream request
        raise AssertionError("Must not reach another request")

    monkeypatch.setattr(evaluation, "main", fake_eval)
    assert runner.judge_worker(SimpleNamespace(name="test", video="study_05")) == 75
    assert calls == [1]
    assert json.loads((directory / "m3_eval_results.jsonl").read_text())["judge_error"]


def test_quota_observation_requires_valid_current_window():
    body = {"base_resp": {"status_code": 0}, "model_remains": [{
        "model_name": "general", "current_interval_remaining_percent": 20,
        "current_weekly_remaining_percent": 80, "end_time": 2000}]}
    assert runner.parse_quota_observation(body, 1000)["interval_remaining_percent"] == 20
    body["model_remains"][0]["current_interval_remaining_percent"] = float("nan")
    with pytest.raises(ValueError):
        runner.parse_quota_observation(body, 1000)
    body["model_remains"][0]["current_interval_remaining_percent"] = 20
    with pytest.raises(ValueError, match="Stale"):
        runner.parse_quota_observation(body, 3000)


def test_quota_query_retries_only_read_only_metadata(monkeypatch):
    import requests
    from unittest.mock import Mock
    monkeypatch.setenv("MINIMAX_API_KEY", "unit-test-key")
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    body = {"base_resp": {"status_code": 0}, "model_remains": [{
        "model_name": "general", "current_interval_remaining_percent": 30,
        "current_weekly_remaining_percent": 90, "end_time": 9999999999}]}
    get = Mock(side_effect=[requests.Timeout("private"), requests.Timeout("private"),
                           SimpleNamespace(status_code=200, json=lambda: body)])
    monkeypatch.setattr(requests, "get", get)
    result = runner.read_quota_observation()
    assert result["read_only_query_attempts"] == 3
    assert result["interval_remaining_percent"] == 30
    assert get.call_count == 3


@pytest.mark.parametrize("available", [False, True])
def test_quota_reserve_stops_before_unattempted_question_without_fake_row(
    tmp_path, monkeypatch, available,
):
    from m3 import qa, config, control_adapter
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "sentinel"))
    monkeypatch.setenv("MINIMAX_API_KEY", "unit-test-key")
    monkeypatch.setenv("OGMEM_GRAPH_CONNECTION_STRING", "test")
    monkeypatch.setattr(runner, "DEADLINE", float("inf"))
    monkeypatch.setattr(config, "load_m3_runtime_config",
                        lambda _: SimpleNamespace(graph_connection_string="test"))
    run = tmp_path / "runs/test"
    run.mkdir(parents=True)
    exp = tmp_path / "experiments/test"
    exp.mkdir(parents=True)
    (exp / "manifest.json").write_text('{"quota_reserve_percent":12}')
    def quota():
        if not available:
            raise TimeoutError("private service details")
        return {"interval_remaining_percent": 11, "weekly_remaining_percent": 90,
                "window_end_unix": 9999999999, "observed_at_unix": 1}
    monkeypatch.setattr(runner, "read_quota_observation", quota)
    original_calls = []
    monkeypatch.setattr(control_adapter.OGMemoryControlAdapter, "run",
                        lambda *args: original_calls.append(1))
    def fake_qa(argv):
        adapter = object.__new__(control_adapter.OGMemoryControlAdapter)
        adapter._llm = SimpleNamespace(_client=SimpleNamespace(api_key="unit-test-key"))
        adapter.run(None, SimpleNamespace(search_stats={}))
        raise AssertionError("Must stop before model execution")
    monkeypatch.setattr(qa, "main_react_qa", fake_qa)
    assert runner.worker(SimpleNamespace(name="test", video="study_05")) == 76
    assert not original_calls
    assert not (run / "study_05/m3_qa_results.jsonl").exists()
    receipt = (run / "study_05/quota_reserve_stop.json").read_text()
    assert "private service details" not in receipt
