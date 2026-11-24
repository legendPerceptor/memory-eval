#!/usr/bin/env python3
"""Re-score CAM answers with the oGMemory 0/0.5/1 judge for fair comparison.

Reads CAM reasoning outputs (data/reasoning/<video>.json), scores every
question with the exact oGMemory judge prompt/schema (MiniMax, temp 0),
normalizes to {0,0.5,1}, and writes per-question + aggregated results.
"""
import json, os, sys, time
from pathlib import Path
from openai import OpenAI

ROOT = Path("/home/yuanjian/Development/memory-eval")
CAM = ROOT / "CAM"
OUT = ROOT / "experiments/cam_paired8_20261123"
VIDEOS = ["study_09","kitchen_01","bedroom_05","study_23","kitchen_23","study_05","gym_03","bedroom_04"]

# --- exact oGMemory judge prompt / schema (oGMemory/m3/eval.py) ---
JUDGE_SCHEMA = {"type":"object","properties":{"score":{"type":"number","description":"1.0 if predicted answer is correct, 0.5 if partially correct, 0.0 if wrong or missing."},"reasoning":{"type":"string","description":"Brief explanation of why the predicted answer received the given score."}},"required":["score","reasoning"]}
JUDGE_PROMPT_TEMPLATE = """You are an exacting evaluator for M3-Bench long-video QA.

The predicted answer and the expected (ground truth) answer are below.
Score the predicted answer:

* 1.0 \u2014 predicted answer is essentially correct (matches expected or
  conveys the same fact, possibly with minor wording differences)
* 0.5 \u2014 predicted answer is partially correct (right idea but wrong
  details, or missing a key part)
* 0.0 \u2014 predicted answer is wrong, contradicts the expected answer,
  or says "insufficient evidence"

Return JSON with keys: score (0/0.5/1) and reasoning.

Question: {question}
Expected answer: {expected}
Predicted answer: {predicted}
"""

def norm(score):
    if score >= 0.75: return 1.0
    if score >= 0.25: return 0.5
    return 0.0

def make_client():
    key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("CAM_LLM_API_KEY")
    if not key:
        raise SystemExit("no MINIMAX_API_KEY")
    base = os.environ.get("CAM_LLM_BASE_URL", "https://api.minimax.cn/v1")
    return OpenAI(api_key=key, base_url=base, timeout=180)

def _is_quota_error(exc):
    text = str(exc)
    code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    return code == 429 or ("用量上限" in text) or ("token plan" in text.lower()) or ("quota" in text.lower())

def judge(client, model, question, expected, predicted):
    if not predicted or not predicted.strip():
        return 0.0, "predicted answer is empty", False
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, expected=expected or "(no expected answer)", predicted=predicted)
    last = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt + "\n\nReturn ONLY a JSON object with keys score and reasoning."}],
                max_tokens=1024, temperature=0.0,
                response_format={"type":"json_object"},
                reasoning_effort="medium",
                extra_body={"thinking":{"type":"adaptive"}},
            )
            raw = resp.choices[0].message.content
            i, j = raw.find("{"), raw.rfind("}")
            obj = json.loads(raw[i:j+1])
            return norm(float(obj.get("score",0.0))), str(obj.get("reasoning","")).strip(), False
        except Exception as e:
            if _is_quota_error(e):
                raise
            last = e
            time.sleep(0.5*(attempt+1))
    return 0.0, f"judge failed: {last}", True

def main():
    model = os.environ.get("CAM_JUDGE_MODEL", os.environ.get("CAM_LLM_MODEL","MiniMax-M3.1-Flash"))
    client = make_client()
    OUT.mkdir(parents=True, exist_ok=True)
    grades_path = OUT / "judge_grades.jsonl"
    summary_path = OUT / "cam_summary.json"

    # ---- Resumable: load already-judged (video, question_id) rows ----
    all_rows = []
    done_keys = set()
    if grades_path.exists():
        for line in grades_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a truncated trailing line from a previous abort
            if r.get("judge_error"):
                continue  # retry errored rows
            done_keys.add((r.get("video_id"), r.get("question_id")))
            all_rows.append(r)
    if done_keys:
        print(f"resume: {len(done_keys)} questions already judged", file=sys.stderr)
    # Rewrite grades with only the clean completed rows (drops stale error rows).
    grades_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + ("\n" if all_rows else "")
    )

    for video in VIDEOS:
        path = CAM / "data" / "reasoning" / f"{video}.json"
        if not path.exists():
            print(f"missing reasoning: {path}", file=sys.stderr); continue
        data = json.loads(path.read_text())
        for qid, rec in data.items():
            if (video, qid) in done_keys:
                continue
            q = rec.get("question","")
            exp = rec.get("ground_truth_answer","")
            reasoning = rec.get("reasoning") or {}
            pred = reasoning.get("final_answer","") if isinstance(reasoning, dict) else ""
            try:
                score, why, err = judge(client, model, q, exp, pred)
            except Exception as e:
                if _is_quota_error(e):
                    print(f"QUOTA during judge at {video} q{qid}; aborting (grades preserved).", file=sys.stderr)
                    sys.exit(3)
                score, why, err = 0.0, f"judge failed: {e}", True
            row = {"video_id":video,"question_id":qid,"question":q,"expected_answer":exp,
                   "predicted_answer":pred,"type":rec.get("type"),"timestamp":rec.get("timestamp"),
                   "judge_score":score,"judge_reasoning":why,"judge_error":err}
            all_rows.append(row)
            # Persist incrementally so a quota abort keeps finished grades.
            with open(grades_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{video} q{qid} -> {score} {'ERR' if err else ''}", file=sys.stderr)

    # ---- Aggregate ----
    by_video = {}
    for video in VIDEOS:
        rows = [r for r in all_rows if r["video_id"] == video and not r["judge_error"]]
        if not rows:
            continue
        by_video[video] = {
            "questions": len(rows),
            "valid_answers": len(rows),
            "mean_score": sum(r["judge_score"] for r in rows)/len(rows),
            "full_credit_fraction": sum(1 for r in rows if r["judge_score"]>=1.0)/len(rows),
        }
    valid_all = [r for r in all_rows if not r["judge_error"]]
    summary = {
        "method":"cam",
        "model":model,
        "judge":"oGMemory 0/0.5/1 (MiniMax, temp 0)",
        "questions":len(all_rows),
        "valid_answers":len(valid_all),
        "judge_errors":sum(1 for r in all_rows if r["judge_error"]),
        "mean_score":(sum(r["judge_score"] for r in valid_all)/len(valid_all)) if valid_all else 0.0,
        "full_credit_fraction":(sum(1 for r in valid_all if r["judge_score"]>=1.0)/len(valid_all)) if valid_all else 0.0,
        "by_video":by_video,
    }
    # Final grades file = clean rows only (preserves canonical ordering).
    grades_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + ("\n" if all_rows else "")
    )
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
