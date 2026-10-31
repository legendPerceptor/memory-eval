#!/usr/bin/env python3
"""Wait for the running driver, then audit and render its actual results.

This never reruns models, changes answers, retries questions, or labels a
terminated/incomplete driver as a finished experiment.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments/m3_flash_medium_20261031"
RUN = ROOT / "runs/m3_flash_medium_20261031"
PYTHON = ROOT / "oGMemory/.venv/bin/python"


def write_status(status, **extra):
    data = {"status": status, "pid": os.getpid(),
            "observed_at": datetime.datetime.now().astimezone().isoformat(), **extra}
    (EXP / "finalization_status.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(data, ensure_ascii=False), flush=True)


def main():
    deadline = time.monotonic() + 24 * 3600
    write_status("waiting_for_driver")
    while not (EXP / "exit_code.txt").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("No driver exit receipt after 24 hours; report not finalized")
        time.sleep(20)
    code = int((EXP / "exit_code.txt").read_text())
    report_file = RUN / "_aggregate/run_report.json"
    if not report_file.exists():
        raise RuntimeError(f"Driver exited {code} without a final run report; not a completed evaluation")
    driver = json.loads(report_file.read_text())
    if driver.get("status") not in ("completed", "failed"):
        raise RuntimeError("Driver report is incomplete; refusing to finalize")
    write_status("auditing", driver_exit_code=code)
    commands = [
        [str(PYTHON), "scripts/summarize_m3.py", "--run", str(RUN), "--output", str(EXP)],
        [str(PYTHON), "scripts/audit_m3_artifacts.py", "--run", str(RUN), "--experiment", str(EXP)],
        [str(PYTHON), "scripts/render_m3_report.py", "--experiment", str(EXP)],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    metrics = json.loads((EXP / "metrics.json").read_text())
    db = json.loads((EXP / "database_audit.json").read_text())
    provenance = json.loads((EXP / "provenance.json").read_text())
    current_patch = subprocess.check_output(["git", "-C", str(ROOT / "oGMemory"), "diff"])
    for name in ("test_reasoning_model_config.py", "test_episodic_workers.py"):
        result = subprocess.run(
            ["git", "-C", str(ROOT / "oGMemory"), "diff", "--no-index", "--", "/dev/null",
             "tests/unit/m3/" + name], capture_output=True, check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError("Could not fingerprint the source patch")
        current_patch += result.stdout
    source_matches = hashlib.sha256(current_patch).hexdigest() == provenance["source_patch_sha256"]
    accounts = [row["account_id"] for row in db["accounts"]]
    integrity = {
        "source_patch_matches_launch": source_matches,
        "expected_account_only": accounts == [provenance["account"]],
        "invalid_edges": db["invalid_edges"][0]["count"],
        "memory_and_index_complete": all(
            v["clips"] > 0 and v["clips"] == v["episodic_records"] == v["index_records"]
            for v in metrics["by_video"].values()),
        "qa_record_coverage_complete": metrics["overall"]["qa_records"] == metrics["overall"]["expected"],
        "judge_errors": metrics["overall"]["judge_errors"],
    }
    provenance.update(
        status="finished", driver_exit_code=code, driver_status=driver["status"],
        finished_audit_at=datetime.datetime.now().astimezone().isoformat(),
        integrity=integrity, report="REPORT.md",
        report_script_sha256={
            name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
            for name in ("summarize_m3.py", "audit_m3_artifacts.py", "render_m3_report.py",
                         "finalize_m3.py", "build_report.sh")
        })
    (EXP / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    warning = (
        "\n\n## 自动收尾校验\n\n"
        f"- 驱动退出码：{code}；真实管线状态：`{driver['status']}`。\n"
        f"- 完整性检查：`{json.dumps(integrity, ensure_ascii=False)}`。\n"
        "- 报告数值由实际结果自动生成；逐题文本审计目前仅覆盖 findings.md 明示的样本。\n"
        "- PDF 构建状态见 finalization_status.json 与 latex/build/build.log；"
        "只有成功编译才发布 latex/report.pdf。\n"
    )
    with (EXP / "REPORT.md").open("a") as f:
        f.write(warning)
    if not source_matches or not integrity["expected_account_only"] or integrity["invalid_edges"]:
        with (ROOT / "latex/results.tex").open("a") as f:
            f.write("\n\\textbf{完整性警告：源码指纹、账号或图关系检查存在异常，须先审查机器审计再解释成绩。}\n")
    write_status("building_pdf", driver_exit_code=code, integrity=integrity)
    subprocess.run(["bash", "scripts/build_report.sh"], cwd=ROOT, check=True)
    provenance["pdf"] = "../../latex/report.pdf"
    provenance["pdf_sha256"] = hashlib.sha256((ROOT / "latex/report.pdf").read_bytes()).hexdigest()
    (EXP / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    readme = ROOT / "README.md"
    text = readme.read_text().replace(
        "正式运行中，4 个独立片段并发",
        "已结束，报告已生成；请核对 REPORT.md 中的失败与完整性检查")
    readme.write_text(text)
    write_status("reports_generated", driver_exit_code=code, integrity=integrity,
                 pdf=str(ROOT / "latex/report.pdf"),
                 mean_score=metrics["overall"]["mean_score"],
                 valid_answers=metrics["overall"]["valid_answers"],
                 expected_questions=metrics["overall"]["expected"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("needs_attention", error_type=type(exc).__name__, error=str(exc))
        raise
