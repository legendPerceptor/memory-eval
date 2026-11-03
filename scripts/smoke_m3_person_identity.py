#!/usr/bin/env python3
"""Read existing perception with an explicit read-only SQL transaction. Never updates it."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oGMemory"))
import psycopg2
from core.graph_enums import RawDataType
from core.graph_models import create_extracted_node, create_raw_node
from core.models import RequestContext
from fs.sql_adapter.pool import PoolAdapterMixin
from m3.person_memory import (
    assign_people,
    build_person_context,
    fingerprint,
    node_key,
    observations_from_graph,
)

parser = argparse.ArgumentParser(
    description="Read-only real-data smoke for video-local identity proposals; no CV or LLM calls."
)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--account", required=True)
parser.add_argument("--video", required=True)
parser.add_argument("--work-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
from m3.config import load_m3_runtime_config

connection_string = load_m3_runtime_config(str(args.config)).graph_connection_string
ctx = RequestContext(
    account_id=args.account,
    user_id="m3bench-user",
    agent_id="m3bench-agent",
    session_id="readonly-person-test",
    trace_id="readonly-person-test",
)
video = args.video
frame_ids = [
    n
    for line in (args.work_dir / "frames.jsonl").read_text().splitlines()
    for n in json.loads(line)["child_node_ids"]
]
started = time.monotonic()
faces = {}
raws = {}
with psycopg2.connect(connection_string) as db:
    db.set_session(readonly=True, isolation_level="REPEATABLE READ")
    with db.cursor() as c:
        PoolAdapterMixin.bind_tenant(db, ctx.account_id)
        c.execute("SHOW transaction_read_only")
        assert c.fetchone()[0] == "on"
        c.execute(
            "SELECT node_id,source_raw_id,account_id,extracted_content FROM graph_nodes WHERE account_id=%s AND extractor_type='face_detection' AND source_raw_id=ANY(%s::uuid[]) AND extracted_content ? 'embedding' ORDER BY node_id",
            (ctx.account_id, frame_ids),
        )
        for nid, parent, account, body in c.fetchall():
            n = create_extracted_node(
                str(parent),
                "face_detection",
                "face_detection_v1",
                account,
                body,
                str(nid),
            )
            faces[node_key(nid)] = n
        frame_ids = list(
            {n.extracted_content["source_frame_node_id"] for n in faces.values()}
        )
        for ids in [frame_ids]:
            c.execute(
                "SELECT node_id,raw_type,content,metadata,account_id FROM graph_nodes WHERE account_id=%s AND node_type='raw' AND node_id=ANY(%s::uuid[])",
                (ctx.account_id, ids),
            )
            for nid, kind, content, meta, account in c.fetchall():
                raws[node_key(nid)] = create_raw_node(
                    RawDataType(kind), content or "", account, meta, str(nid)
                )
        clip_ids = list({n.metadata["source_video_node_id"] for n in raws.values()})
        c.execute(
            "SELECT node_id,raw_type,content,metadata,account_id FROM graph_nodes WHERE account_id=%s AND node_type='raw' AND node_id=ANY(%s::uuid[])",
            (ctx.account_id, clip_ids),
        )
        for nid, kind, content, meta, account in c.fetchall():
            raws[node_key(nid)] = create_raw_node(
                RawDataType(kind), content or "", account, meta, str(nid)
            )
        c.execute(
            "SELECT count(*) FROM graph_nodes WHERE account_id=%s AND extractor_type='face_detection' AND extracted_content ? 'embedding'",
            (ctx.account_id,),
        )
        account_faces = c.fetchone()[0]
# Legacy CV children lack external video/clip tags. Reconstruct them in this
# in-memory read-only view exactly as the new detector cache-resume path does.
for n in faces.values():
    frame = raws[node_key(n.extracted_content["source_frame_node_id"])]
    clip = raws[node_key(frame.metadata["source_video_node_id"])]
    assert clip.metadata["video_id"] == video
    n.extracted_content = {
        **n.extracted_content,
        "source_video_id": video,
        "source_clip_id": clip.metadata["clip_id"],
    }
assert len(faces) > 0
loaded = time.monotonic()


class Graph:
    def list_extracted_nodes(self, kind, request, *, limit, video_id):
        assert (
            kind == "face_detection"
            and video_id == video
            and request.account_id == ctx.account_id
        )
        return list(faces.values())

    def get_raw_node(self, nid, request):
        return raws.get(node_key(nid))

    def get_extracted_node(self, nid, request):
        return faces.get(node_key(nid))


g = Graph()
obs = observations_from_graph(g, ctx, video)
observed = time.monotonic()
manifest = assign_people(obs, ctx.account_id, video)
matched = time.monotonic()
prefix = [o for o in obs if o["clip_id"] < 10]
p = assign_people(prefix, ctx.account_id, video)
assert p["observations"] == manifest["observations"][: len(prefix)]
crops = []
for clip_id in sorted({o["clip_id"] for o in obs})[
    :: max(1, len({o["clip_id"] for o in obs}) - 1)
]:
    raw = next(
        n
        for n in raws.values()
        if n.raw_type == RawDataType.VIDEO and n.metadata["clip_id"] == clip_id
    )
    frames = [
        n
        for n in raws.values()
        if n.raw_type == RawDataType.IMAGE
        and node_key(n.metadata["source_video_node_id"]) == node_key(raw.node_id)
    ]
    context = build_person_context(manifest, raw, frames, g, ctx)
    crops.append(
        {
            "clip": clip_id,
            "crops": len(context["entries"]),
            "context_sha256": fingerprint(context),
        }
    )
result = {
    "video": video,
    "legacy_provenance_reconstructed_in_memory": True,
    "read_only_transaction": True,
    "database_writes": 0,
    "model_calls": 0,
    "account_faces": account_faces,
    "selected_faces": len(faces),
    "valid_observations": len(obs),
    "all_proposal_ids": len({o["person_id"] for o in manifest["observations"]}),
    "resolved_track_proposals": len(
        {o["person_id"] for o in manifest["observations"] if not o["ambiguous_match"]}
    ),
    "ambiguous_observations": sum(
        o["ambiguous_match"] for o in manifest["observations"]
    ),
    "load_seconds": loaded - started,
    "observation_validation_seconds": observed - loaded,
    "causal_matching_seconds": matched - observed,
    "prefix_observations_checked": len(prefix),
    "prefix_assignments_unchanged": True,
    "crop_checks": crops,
    "manifest_sha256": fingerprint(manifest),
    "warning": "New causal matching differs from old batch single-linkage; this is not a QA accuracy experiment.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
