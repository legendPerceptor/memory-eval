#!/usr/bin/env python3
"""Make a single small request to check whether the evaluation API has recovered."""

import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments/m3_early_answer_20261109"
os.environ["M3_USAGE_LOG"] = str(EXP / "usage-quota-probes.jsonl")
logging.disable(logging.CRITICAL)
from m3.eval import build_judge_llm, judge_answer

llm = build_judge_llm(config_path=str(EXP / "private/moderate-bedroom_04.yaml"))
score, reason, error = judge_answer(judge_llm=llm, question="What is two plus two?",
                                    expected="Four", predicted="Four")
category = ("plan_usage_limit" if "2056" in reason else
            "rate_limit" if "2062" in reason or "429" in reason else
            "other_error" if error else "available")
record = {"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "category": category,
          "score": score if not error else None}
with (EXP / "quota-probes.jsonl").open("a") as stream:
    stream.write(json.dumps(record) + "\n")
print(json.dumps(record))
sys.exit(1 if error else 0)
