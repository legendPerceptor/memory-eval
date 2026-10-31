#!/usr/bin/env python3
"""Replay a real failed clip, changing only reasoning effort; no memory writes."""
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oGMemory"))
from core.models import RequestContext
from extraction.operators.builtin.episodic_memory_generator import EpisodicMemoryGenerator
from m3.config import load_m3_runtime_config
from m3.episodic_memory import _index_asr_segments, _index_audio_records, _index_frame_records
from m3.load_clips import build_graph_service
from providers.llm.minimax_llm import MiniMaxLLM

exp = ROOT / "experiments/m3_minimax_flash_20261031"
source = ROOT / "runs/m3_minimax_flash_20261031/study_05"
runtime = load_m3_runtime_config(str(exp / "private/config.yaml"))
graph, _ = build_graph_service("sql", runtime.graph_connection_string)
ctx = RequestContext(account_id="m3-flash-20261031", user_id="m3bench-user",
                     agent_id="m3bench-agent", session_id="diagnostic", trace_id="diagnostic")
frames = _index_frame_records(source / "frames.jsonl")
audio = _index_audio_records(source / "audio.jsonl")
asr = _index_asr_segments(graph, ctx, audio)
manifest = json.loads((source / "data.jsonl").read_text().splitlines()[0])
raw = next(r for r in manifest["raw_nodes"] if r["clip_id"] == 0)
key = "study_05|0"
llm = MiniMaxLLM(
    api_key=os.environ["MINIMAX_API_KEY"], base_url="https://api.minimax.cn/v1",
    model="MiniMax-M3.1-Flash", max_tokens=8192, request_timeout_seconds=300,
    is_reasoning_model=True, reasoning_effort="medium",
    extra_body={"thinking": {"type": "adaptive"}},
)
result = EpisodicMemoryGenerator(llm=llm, max_frames=16, include_audio=True).extract(
    graph.get_raw_node(raw["raw_node_id"], ctx), ctx,
    frame_nodes=[graph.get_raw_node(n, ctx) for n in frames[key]],
    audio_node=graph.get_raw_node(audio[key], ctx), asr_segments=asr[key])
(exp / "private/medium-same-input-diagnostic.json").write_text(
    json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
print(json.dumps({"error": result.error, "frame_count": result.extracted_content.get("frame_count"),
                  "audio_attached": result.extracted_content.get("audio_attached"),
                  "description": result.extracted_content.get("description")}, ensure_ascii=False))
raise SystemExit(2 if result.error else 0)
