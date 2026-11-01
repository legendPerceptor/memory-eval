import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import build_prefix, crop_image, anchor_update, Controller, Task, ROOT
from reference import load_reference, normalise_memory
from core.models import RequestContext


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.ref = load_reference(ROOT / "m3-agent", lambda model, texts: ([[1., 0.] for _ in texts], 0))
        self.updates = [{"clip_id": 0, "updates": [
            {"id": 0, "type": "img", "info": {"embeddings": [[1., 0.]], "contents": ["crop"]}},
            {"id": 1, "type": "voice", "info": {"embeddings": [[1., 0.]], "contents": ["speaker"]}},
        ]}]
        self.memories = {
            0: {"video_descriptions": ["<face_0> drinks tea."],
                "high_level_conclusions": ["Equivalence: <face_0>, <voice_1>", "<voice_1> is Alice."]},
            1: {"video_descriptions": ["<face_0> leaves."],
                "high_level_conclusions": ["<voice_1> is Alice."]},
        }

    def test_original_graph_equivalence_is_used(self):
        g = build_prefix(self.ref, self.updates, self.memories, 0, 2)
        self.assertEqual(g.reverse_character_mappings["face_0"], g.reverse_character_mappings["voice_1"])
        found, seen, scores = self.ref.search(g, "tea", [], topk=2, threshold=0.5, before_clip=0)
        self.assertEqual(list(found), ["CLIP_0"])
        self.assertTrue(any("character_" in s for s in found["CLIP_0"]))
        self.assertFalse(any(s.startswith("Equivalence:") for s in found["CLIP_0"]))

    def test_prefix_excludes_future_text_and_reinforcement(self):
        early = build_prefix(self.ref, self.updates, self.memories, 0, 2)
        late = build_prefix(self.ref, self.updates, self.memories, 1, 2)
        self.assertNotIn(1, early.text_nodes_by_clip)
        self.assertIn(1, late.text_nodes_by_clip)
        self.assertGreater(sum(late.edges.values()), sum(early.edges.values()))
        repeated = build_prefix(self.ref, self.updates, self.memories, 0, 2)
        self.assertEqual(early.edges, repeated.edges)

    def test_singular_output_key_normalized_without_inventing_content(self):
        r = normalise_memory({"video_description": ["fact"], "high_level_conclusions": []})
        self.assertEqual(r, {"video_descriptions": ["fact"], "high_level_conclusions": []})

    def test_bad_memory_rejected(self):
        for value in [{}, {"name": "Alice"}, {"video_descriptions": [], "high_level_conclusions": []},
                      {"video_description": ["x"], "video_descriptions": ["y"], "high_level_conclusions": []}]:
            with self.assertRaises(ValueError):
                normalise_memory(value)

    def test_controller_runs_actual_search_and_captures_answer(self):
        class LLM:
            def __init__(self):
                self.calls = 0
            def complete_with_tools_messages(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return [{"tool": "Search", "input": {"query": "tea"}, "id": "one"}], ""
                assert any("CLIP_0" in str(m.get("content")) for m in messages)
                return [{"tool": "Answer", "input": {"final_answer": "Tea"}, "id": "two"}], ""
        graph = build_prefix(self.ref, self.updates, self.memories, 0, 2)
        task = Task("What does Alice drink?")
        adapter = Controller(LLM(), self.ref, graph, 0, {"failed_clips": []})
        ctx = RequestContext(account_id="unit", user_id="u", agent_id="a", session_id="s", trace_id="t")
        result = adapter.run(ctx, task)
        self.assertTrue(result.completed)
        self.assertEqual(result.output, ["Tea"])
        self.assertEqual(len(task.search_history), 1)

    def test_duplicate_clips_not_returned_again(self):
        g = build_prefix(self.ref, self.updates, self.memories, 0, 2)
        _, seen, _ = self.ref.search(g, "tea", [], topk=2)
        found, _, _ = self.ref.search(g, "drink", seen, topk=2)
        self.assertEqual(found, {})

    def test_normalized_face_box_is_scaled_before_cropping(self):
        import base64
        from io import BytesIO
        from PIL import Image
        buffer = BytesIO()
        Image.new("RGB", (128, 64), "red").save(buffer, format="JPEG")
        raw = SimpleNamespace(content=buffer.getvalue(), metadata={})
        result = Image.open(BytesIO(base64.b64decode(crop_image(raw, [.25, .25, .75, .75]))))
        self.assertEqual(result.size, (64, 32))

    def test_anchor_journal_does_not_alias_upstream_mutable_contents(self):
        graph = self.ref.VideoGraph(**self.ref.memory_config)
        info = {"embeddings": [[1., 0.]], "contents": ["first"]}
        node = graph.add_img_node(info)
        recorded = anchor_update(node, "img", info)
        graph.update_node(node, {"embeddings": [[1., 0.]], "contents": ["later"]})
        assert info["contents"] == ["first", "later"]
        assert recorded["info"]["contents"] == ["first"]

    def test_completed_run_is_rejected_before_any_preparation(self):
        import tempfile
        from unittest.mock import patch
        import runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "v").mkdir()
            (root / "v/m3_qa_results.jsonl").write_text("existing\n")
            with patch.object(runner, "RUN", root):
                with self.assertRaisesRegex(RuntimeError, "QA output exists"):
                    runner.run_video("v", root / "nonexistent-config.yaml")


if __name__ == "__main__":
    unittest.main()
