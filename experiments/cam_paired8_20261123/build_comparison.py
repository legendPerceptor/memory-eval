#!/usr/bin/env python3
"""Build CAM vs oGMemory(latest paper) vs m3_agent(frozen) comparison.

oGMemory uses the latest 4-way parallel re-run in the paper
(experiments/m3_parallel4_paper8_20261113/judgments.jsonl, 97/97, 63.92%),
NOT the older summary in m3_paper_paired8_20261103/comparison.json.
m3_agent stays frozen from m3_paper_paired8_20261103/comparison.json.
"""
import json
from collections import defaultdict
from pathlib import Path
ROOT = Path("/home/yuanjian/Development/memory-eval")
OUT = ROOT / "experiments/cam_paired8_20261123"
base = json.loads((ROOT/"experiments/m3_paper_paired8_20261103/comparison.json").read_text())
cam = json.loads((OUT/"cam_summary.json").read_text())
vids = base["cohort"]

def _load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

og_rows = _load_jsonl(ROOT/"experiments/m3_parallel4_paper8_20261113/judgments.jsonl")
og_byv = defaultdict(list)
for r in og_rows:
    if r.get("judge_error"):
        continue
    og_byv[r["video_id"]].append(float(r["score"]))
og_all = [x for xs in og_byv.values() for x in xs]
ogsummary = {
    "questions": len(og_rows),
    "valid_answers": len(og_all),
    "mean_score": sum(og_all)/len(og_all),
    "full_credit_fraction": sum(1 for x in og_all if x >= 1.0)/len(og_all),
    "by_video": {v: {"questions": len(og_byv[v]), "valid_answers": len(og_byv[v]),
                     "mean_score": sum(og_byv[v])/len(og_byv[v]),
                     "full_credit_fraction": sum(1 for x in og_byv[v] if x >= 1.0)/len(og_byv[v])}
                 for v in og_byv},
}

rows = []
def add(name, d):
    rows.append({"method":name,"questions":d.get("questions"),"valid_answers":d.get("valid_answers"),
                 "mean_score":round(d.get("mean_score",0),4),
                 "full_credit_fraction":round(d.get("full_credit_fraction",0),4)})
add("CAM (this run)", cam)
add("ogmemory(latest)", ogsummary)
add("m3_agent(frozen)", base["methods"]["m3_agent"])
print("overall:")
print(f"{'method':20s} {'n':>4s} {'valid':>5s} {'mean':>7s} {'full':>7s}")
for r in rows:
    print(f"{r['method']:20s} {r['questions']:>4} {r['valid_answers']:>5} {r['mean_score']:>7.4f} {r['full_credit_fraction']:>7.4f}")
print("\nby video (mean_score):")
print(f"{'video':12s} {'CAM':>7s} {'oGMem':>7s} {'m3agt':>7s}")
byv = {"CAM":cam.get("by_video",{}), "ogmemory":ogsummary["by_video"],
       "m3_agent":base["methods"]["m3_agent"].get("by_video",{})}
for v in vids:
    def g(n):
        x = byv.get(n,{}).get(v,{}); return x.get("mean_score")
    c=g("CAM"); o=g("ogmemory"); a=g("m3_agent")
    f=lambda x: f"{x:.4f}" if isinstance(x,(int,float)) else "  -   "
    print(f"{v:12s} {f(c):>7s} {f(o):>7s} {f(a):>7s}")
json.dump({"rows":rows,"by_video":byv,
           "note":"oGMemory = latest 4-way parallel paper re-run (m3_parallel4_paper8_20261113); m3_agent frozen."},
          open(OUT/"comparison_cam.json","w"), ensure_ascii=False, indent=2)
