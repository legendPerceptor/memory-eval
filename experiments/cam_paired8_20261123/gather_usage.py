#!/usr/bin/env python3
"""Aggregate CAM token usage + wall-clock time per video and overall.
Reads memory_token_summaries from CAM/data/memorization/<v>.json and
per-question token_summaries from CAM/data/reasoning/<v>.json, and elapsed
times parsed from CAM/data/logs/<v>_memo.log and *_reason_*.log."""
import json, re, glob
from pathlib import Path
CAM = Path("/home/yuanjian/Development/memory-eval/CAM")
OUT = Path("/home/yuanjian/Development/memory-eval/experiments/cam_paired8_20261123")
VIDEOS = ["study_09","kitchen_01","bedroom_05","study_23","kitchen_23","study_05","gym_03","bedroom_04"]

def memo_tokens(v):
    p = CAM/f"data/memorization/{v}.json"
    if not p.exists(): return None
    try:
        d=json.loads(p.read_text()); return int(d.get("memory_token_summaries",{}).get("total",0))
    except Exception: return None

def reason_tokens(v):
    p = CAM/f"data/reasoning/{v}.json"
    if not p.exists(): return None
    try:
        d=json.loads(p.read_text()); t=0; n=0
        for qid,rec in d.items():
            rs=rec.get("reasoning")
            if isinstance(rs,dict):
                t+=int(rs.get("token_summaries",{}).get("total",0) or 0); n+=1
        return t,n
    except Exception: return None

def _scan_for_elapsed(paths, v, kind):
    best=None
    for lg in paths:
        try: txt=Path(lg).read_text(errors="ignore")
        except Exception: continue
        # tqdm uses \r; normalise then search
        for m in re.finditer(rf"\[{re.escape(v)}\]\s+{kind}\s+(\d+)s", txt):
            best=int(m.group(1))
    return best

def parse_elapsed(v, kind):
    # The "✓ [v] memo/reason Ns" summary line goes to the per-run stdout file
    # (/tmp/<v>_<kind>.out), written by the supervisor; fall back to logs.
    if kind=="memo":
        cand=[Path(f"/tmp/{v}_memo.out"), CAM/f"data/logs/{v}_memo.log"]
    else:
        cand=[Path(f"/tmp/{v}_reason.out")]+[Path(x) for x in sorted(glob.glob(str(CAM/f"data/logs/{v}_reason_*.log")))]
    return _scan_for_elapsed(cand, v, kind)

rows={}; tot={"memo_tokens":0,"reason_tokens":0,"memo_s":0,"reason_s":0,"reason_q":0}
for v in VIDEOS:
    mt=memo_tokens(v); rt=reason_tokens(v)
    ms=parse_elapsed(v,"memo"); rs=parse_elapsed(v,"reason")
    row={"memo_tokens":mt,
         "reason_tokens":(rt[0] if rt else None),
         "reason_answered":(rt[1] if rt else None),
         "memo_seconds":ms,"reason_seconds":rs}
    rows[v]=row
    if mt: tot["memo_tokens"]+=mt
    if rt: tot["reason_tokens"]+=rt[0]; tot["reason_q"]+=rt[1]
    if ms: tot["memo_s"]+=ms
    if rs: tot["reason_s"]+=rs

summary={"by_video":rows,"totals":tot,
         "total_tokens":tot["memo_tokens"]+tot["reason_tokens"],
         "allocated_seconds_serial":tot["memo_s"]+tot["reason_s"]}
OUT.mkdir(parents=True, exist_ok=True)
(OUT/"cam_usage.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2,ensure_ascii=False))
