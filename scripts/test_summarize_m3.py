import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from summarize_m3 import summarize


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        (self.run / "v").mkdir(parents=True)
        (self.run / "_aggregate").mkdir()
        self.ann = {"v": {"qa_list": [
            {"question_id": "q1", "question": "First?", "answer": "A", "type": ["spatial"]},
            {"question_id": "q2", "question": "Second?", "answer": "B", "type": ["count"]},
        ]}}
        self.annotations = self.root / "annotations.json"
        self.annotations.write_text(json.dumps(self.ann))
        self.manifest = {
            "selected_video_ids": ["v"], "total_questions": 2,
            "annotations_sha256": hashlib.sha256(
                json.dumps(self.ann, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        }
        (self.run / "sample_manifest.json").write_text(json.dumps(self.manifest))

    def write_rows(self, qa, judged=()):
        (self.run / "v/m3_qa_results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in qa))
        (self.run / "_aggregate/m3_eval_results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in judged))

    def summary(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return summarize(self.run, self.annotations, self.root / "out")

    def test_missing_questions_stay_in_denominator(self):
        a = {"video_id": "v", "question_id": "q1", "predicted_answer": "A"}
        self.write_rows([a], [{**a, "judge_score": 1}])
        b = self.summary()["overall"]
        self.assertEqual(b["mean_score"], 0.5)
        self.assertEqual(b["missing_qa"], 1)
        self.assertEqual(b["expected"], 2)

    def test_judge_error_is_unknown_not_model_wrong(self):
        a = {"video_id": "v", "question_id": "q1", "predicted_answer": "A"}
        self.write_rows([a], [{**a, "judge_score": 0, "judge_error": True}])
        b = self.summary()["overall"]
        self.assertIsNone(b["mean_score"])
        self.assertEqual(b["judge_errors"], 1)

    def test_nonempty_failed_answer_is_not_success(self):
        a = {"video_id": "v", "question_id": "q1", "predicted_answer": "Answer(...)", "error": "protocol"}
        self.write_rows([a])
        b = self.summary()["overall"]
        self.assertEqual(b["valid_answers"], 0)
        self.assertEqual(b["qa_failures"], 1)

    def test_reject_duplicate_attempts(self):
        a = {"video_id": "v", "question_id": "q1", "predicted_answer": "A"}
        self.write_rows([a, a])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.summary()

    def test_reject_stale_judging(self):
        a = {"video_id": "v", "question_id": "q1", "predicted_answer": "A"}
        self.write_rows([a], [{**a, "predicted_answer": "B", "judge_score": 1}])
        with self.assertRaisesRegex(ValueError, "differs from QA"):
            self.summary()

    def test_reject_changed_annotation(self):
        self.annotations.write_text(json.dumps({"v": {"qa_list": []}}))
        with self.assertRaisesRegex(ValueError, "annotations differ"):
            self.summary()

    def test_reject_out_of_cohort(self):
        self.write_rows([{"video_id": "v", "question_id": "unexpected", "predicted_answer": "A"}])
        with self.assertRaisesRegex(ValueError, "outside"):
            self.summary()

    def test_video_manifest_contains_many_clips(self):
        (self.run / "v/data.jsonl").write_text(json.dumps(
            {"id": "v", "raw_nodes": [{"clip_id": 0}, {"clip_id": 1}]}) + "\n")
        self.assertEqual(self.summary()["by_video"]["v"]["clips"], 2)

    def test_multilabel_counts_do_not_change_denominator(self):
        self.ann["v"]["qa_list"][0]["type"] = ["spatial", "count"]
        self.annotations.write_text(json.dumps(self.ann))
        self.manifest["annotations_sha256"] = hashlib.sha256(
            json.dumps(self.ann, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        (self.run / "sample_manifest.json").write_text(json.dumps(self.manifest))
        result = self.summary()
        self.assertEqual(result["overall"]["expected"], 2)
        self.assertEqual(result["by_label_overlapping"]["count"]["expected"], 2)
        self.assertEqual(result["by_label_overlapping"]["spatial"]["expected"], 1)


if __name__ == "__main__":
    unittest.main()
