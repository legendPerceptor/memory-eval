#!/usr/bin/env python3
"""Real multimodal/operator + tool-call preflight; never writes benchmark memory."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oGMemory"))

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators.builtin.episodic_memory_generator import EpisodicMemoryGenerator
from m3.llm_factory import build_episodic_llm, build_qa_llm

config = str(ROOT / "experiments/m3_minimax_flash_20261031/private/config.yaml")
source = ROOT / "runs/m3_baseline_20261031/bedroom_04"
ctx = RequestContext(account_id="preflight", user_id="preflight", agent_id="preflight",
                     session_id="preflight", trace_id="preflight")
raw = create_raw_node(raw_type=RawDataType.VIDEO, account_id="preflight", content=b"",
                      metadata={"video_id": "bedroom_04", "clip_id": 0,
                                "start_sec": 0, "duration_sec": 30})
paths = sorted((source / "frames/bedroom_04/0").glob("*.jpg"), key=lambda p: int(p.stem))
assert len(paths) == 16, len(paths)
frames = [create_raw_node(raw_type=RawDataType.IMAGE, account_id="preflight",
                          content=p.read_bytes()) for p in paths]
audio = create_raw_node(raw_type=RawDataType.AUDIO, account_id="preflight",
                        content=(source / "audio/bedroom_04/0.wav").read_bytes())
segments = [json.loads(s) for s in (source / "voice_asr.jsonl").read_text().splitlines()]
segments = [s for s in segments if s.get("clip_id") == 0]
llm = build_episodic_llm(config_path=config)
result = EpisodicMemoryGenerator(llm=llm, max_frames=16, include_audio=True).extract(
    raw, ctx, frame_nodes=frames, audio_node=audio, asr_segments=segments)
out = ROOT / "experiments/m3_minimax_flash_20261031/private"
(out / "multimodal-preflight.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
print("Multimodal preflight:", json.dumps({
    "error": result.error, "frames": result.extracted_content.get("frame_count"),
    "audio_attached": result.extracted_content.get("audio_attached"),
    "description": result.extracted_content.get("description"),
}, ensure_ascii=False))
if result.error:
    raise SystemExit(2)
qa = build_qa_llm(config_path=config)
tools = [{"name": "Answer", "description": "Return your answer",
          "input_schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                           "required": ["answer"]}}]
calls = qa.complete_with_tools("Use Answer to state which is larger: 9.11 or 9.9.", tools,
                               tool_choice="required")
(out / "tools-preflight.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2))
assert calls and calls[0]["tool"] == "Answer", calls
print("Native tool preflight:", json.dumps(calls, ensure_ascii=False))
