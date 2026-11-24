#!/usr/bin/env python3
"""One-glance status of the CAM paired-8 run (robust process counting)."""
import json, os, pickle, sys, time
from pathlib import Path

CAM = Path("/home/yuanjian/Development/memory-eval/CAM")
sys.path.insert(0, str(CAM))  # so pickle can import CAM classes/ when run from repo root
VIDEOS = ["study_09","kitchen_01","bedroom_05","study_23","kitchen_23","study_05","gym_03","bedroom_04"]

def scan_procs():
    """Return counts by inspecting /proc/<pid>/cmdline (no self-match issues)."""
    mem = rea = orch = sup = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cl = open(f"/proc/{d}/cmdline","rb").read().replace(b"\0", b" ").decode("utf-8","ignore")
        except Exception:
            continue
        if "bash -c" in cl or "pgrep" in cl or "grep" in cl:
            continue
        if "process_full_video.py" in cl and "python" in cl:
            mem += 1
        elif "reason.py" in cl and "python" in cl:
            rea += 1
        elif ("run_full_job.sh" in cl or "supervisor_pervideo.sh" in cl) and "bash" in cl and "bash -c" not in cl:
            sup += 1
        elif "orchestrator.sh" in cl:
            orch += 1
    return orch, sup, mem, rea

def load_count(p):
    for _ in range(8):
        try:
            with open(p,"rb") as f:
                return len(pickle.load(f)["done_clips"])
        except Exception:
            time.sleep(0.3)
    return -1

td=tn=0; rows=[]
for v in VIDEOS:
    p = CAM/f"data/graphs/{v}_checkpoint.pkl"
    fdir = CAM/f"data/frames/{v}"
    n = len(os.listdir(fdir)) if fdir.is_dir() else 0
    g = (CAM/f"data/graphs/{v}.pkl").exists()
    # A finished graph has no checkpoint left; count it as fully built.
    d = n if g else (load_count(p) if p.exists() else 0)
    rj = CAM/f"data/reasoning/{v}.json"
    nq = 0
    if rj.exists():
        try: nq = len(json.loads(rj.read_text()))
        except Exception: nq = -1
    rd = (CAM/f"data/reasoning/{v}.done").exists()
    td += max(d,0); tn += n
    rows.append((v,d,n,g,nq,rd))

orch, sup, mem, rea = scan_procs()
print("time:", time.strftime("%F %T"))
print(f"orchestrator={'UP' if orch else 'DOWN'}  supervisor_proc={sup}  mem_workers={mem}  reason_workers={rea}")
print(f"{'video':12s} {'clips':>9s} {'graph':>5s} {'q_done':>6s} {'reason_done':>11s}")
for v,d,n,g,nq,rd in rows:
    print(f"{v:12s} {('%d/%d'%(d,n)):>9s} {('Y' if g else '-'):>5s} {str(nq):>6s} {('YES' if rd else '-'):>11s}")
print((f"BUILD TOTAL: {td}/{tn} = {100*td/tn:.1f}%" if tn else "BUILD TOTAL: n/a"))
