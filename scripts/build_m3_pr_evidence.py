#!/usr/bin/env python3
"""Build PR evidence from frozen local experiment results; no model/network calls."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SOURCES = {
    "context": "experiments/m3_context_study_20261101/comparison.json",
    "facts": "experiments/m3_memory_design_20261102/comparison.json",
    "incremental_audit": "experiments/m3_memory_design_20261102/strict_delivery_audit.json",
    "ten_grades": "experiments/m3_ten_fullinterval_20261102/joint_questions.jsonl",
    "person": "docs/m3_person_grounding_implementation_20261103.json",
}


def build():
    inputs = {key: ROOT / path for key, path in SOURCES.items()}
    context = json.loads(inputs["context"].read_text())["methods"]
    facts = json.loads(inputs["facts"].read_text())["methods"]
    person = json.loads(inputs["person"].read_text())
    grades = [
        json.loads(line)
        for line in inputs["ten_grades"].read_text().splitlines()
        if line.strip()
    ]
    videos = [
        "study_09",
        "kitchen_01",
        "bedroom_05",
        "study_23",
        "kitchen_23",
        "study_05",
        "gym_03",
        "bedroom_04",
        "study_18",
        "kitchen_20",
    ]

    def extract(method):
        return {k: method[k] for k in ("mean_score", "valid_answers", "qa_usage")}

    first = {
        key: extract(context[key])
        for key in ("historical_ogmemory", "finalization", "finalization_repeat")
    }
    second = {
        key: extract(facts[key])
        for key in (
            "balanced_contemporary",
            "strict_incremental",
            "m3_reference_repeat",
        )
    }
    a = first["historical_ogmemory"]["qa_usage"]
    b = first["finalization"]["qa_usage"]
    c = second["balanced_contemporary"]["qa_usage"]
    e = second["strict_incremental"]["qa_usage"]
    assert (
        a["prompt_tokens"],
        b["prompt_tokens"],
        c["prompt_tokens"],
        e["prompt_tokens"],
    ) == (19905749, 4804362, 4721119, 2915704)
    assert all(
        m["qa_usage"]["unknown_usage_calls"] == 0
        for m in [*first.values(), *second.values()]
    )
    assert (
        len(grades) == 256
        and len({(r["method"], r["video_id"], r["question_id"]) for r in grades}) == 256
    )
    assert not any(r["judge_error"] for r in grades)

    def metrics(rows):
        return {
            "questions": len(rows),
            "mean_score": sum(r["judge_score"] for r in rows) / len(rows),
            "valid_answers": sum(
                not r["error"] and bool(r["predicted_answer"]) for r in rows
            ),
        }

    ten = {}
    for method in sorted({r["method"] for r in grades}):
        selected = [r for r in grades if r["method"] == method]
        ten[method] = {
            "all_128": metrics(selected),
            "first_eight_97": metrics(
                [r for r in selected if r["video_id"] in videos[:8]]
            ),
            "by_video": {
                v: metrics([r for r in selected if r["video_id"] == v]) for v in videos
            },
        }
    smoke = person["read_only_real_data_smoke"]

    def revision(ref):
        return subprocess.check_output(
            ["git", "-C", str(ROOT / "oGMemory"), "rev-parse", ref], text=True
        ).strip()

    result = {
        "source_branch": "m3_dev_1031",
        "source_commit": revision("m3_dev_1031"),
        "target_branch": "dev",
        "target_snapshot": revision("origin/dev"),
        "experiment_sources": {
            key: {
                "path": SOURCES[key],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in inputs.items()
        },
        "study_1": first,
        "study_2": second,
        "ten_video_final_judge": ten,
        "reductions": {
            "study_1_prompt_fraction": 1 - b["prompt_tokens"] / a["prompt_tokens"],
            "study_1_peak_fraction": 1
            - b["peak_prompt_tokens"] / a["peak_prompt_tokens"],
            "study_2_prompt_fraction": 1 - e["prompt_tokens"] / c["prompt_tokens"],
            "study_2_uncached_fraction": 1
            - e["uncached_prompt_tokens"] / c["uncached_prompt_tokens"],
            "person_selected_face_fraction_reduction": 1
            - smoke["selected_faces"] / smoke["account_faces"],
        },
        "strict_incremental_audit": json.loads(inputs["incremental_audit"].read_text()),
        "person_readonly_smoke": smoke,
        "caveats": [
            "Two independent judge batches: never splice the higher scores across studies.",
            "QA-only fixed-memory studies, not complete-PR head-to-head end-to-end ablations.",
            "Ten-video source 8d61b772 excludes latest person grounding and has infrastructure failures.",
            "No new-person QA gain or concurrency-branch speedup is attributed to this PR.",
            "Input tokens are not total pipeline tokens, billed cost or a runtime speedup.",
        ],
    }
    target = DOCS / "m3_dev_1031_pr_evidence.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    values = [
        first[k]["qa_usage"]["prompt_tokens"] / 1e6
        for k in ("historical_ogmemory", "finalization", "finalization_repeat")
    ]
    bars = axes[0].bar(
        ["Original QA", "Balanced", "Balanced repeat"],
        values,
        color=["#94a3b8", "#2563eb", "#60a5fa"],
        width=0.6,
    )
    axes[0].bar_label(bars, fmt="%.2fM", padding=3)
    axes[0].set_title(
        "Study 1: bounded context + final answer\n5 videos / 65 questions"
    )
    axes[0].set_ylabel("QA prompt tokens (millions)")
    axes[0].set_ylim(0, 23)
    labels = [
        "Balanced\ncontemporary",
        "Strict incremental\nE",
        "M3-Agent\nreference repeat",
    ]
    keys = ["balanced_contemporary", "strict_incremental", "m3_reference_repeat"]
    uncached = [second[k]["qa_usage"]["uncached_prompt_tokens"] / 1e6 for k in keys]
    cached = [second[k]["qa_usage"]["cached_prompt_tokens"] / 1e6 for k in keys]
    axes[1].bar(labels, uncached, color="#2563eb", label="Uncached", width=0.6)
    bars = axes[1].bar(
        labels, cached, bottom=uncached, color="#93c5fd", label="Cached", width=0.6
    )
    axes[1].bar_label(
        bars, labels=[f"{x + y:.2f}M" for x, y in zip(uncached, cached)], padding=3
    )
    axes[1].set_title(
        "Study 2: source facts + strict incremental delivery\nSame 5 development videos / 65 questions"
    )
    axes[1].set_ylim(0, 5.7)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    folder = DOCS / "figures"
    folder.mkdir(exist_ok=True)
    figure = folder / "m3_dev_1031_pr_tokens.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "evidence": str(target.relative_to(ROOT)),
                "figure": str(figure.relative_to(ROOT)),
                "reductions": result["reductions"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    build()
