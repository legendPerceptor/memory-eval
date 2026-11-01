import hashlib

import replay_grouped_evidence as replay


def content():
    return {
        "source_start_sec": 0, "source_duration_sec": 10, "source_end_sec": 10,
        "source_video_id": "v", "source_clip_id": 0,
        "description": "Object on desk.",
        "speech_transcript": [
            {"start_sec": 0, "end_sec": 3, "text": "Safe early words.", "speaker_id": "s"},
            {"start_sec": 4, "end_sec": 8, "text": "Future words.", "speaker_id": "s"},
        ],
    }


def source(body):
    return {"id": "e1", "node_id": "n", "video_id": "v", "clip_id": 0,
            "content_sha256": hashlib.sha256(replay.dumps(body).encode()).hexdigest(),
            "ranges": [{"field": "speech_transcript", "index": 0}]}


def test_replay_checks_source_hash_and_preserves_values():
    c = content()
    body = replay.body_for(c, {"node_id": "n", "video_id": "v", "clip_id": 0})
    doc, boundary = replay.restore_document(source(body), c, 10)
    assert not boundary
    assert doc["passages"][0]["value"] == c["speech_transcript"][0]
    grouped = replay.model_document(doc, "grouped")
    replay.verify_grouped(doc, grouped)


def test_boundary_replay_does_not_restore_future_or_speaker_identity():
    c = content()
    projected = replay.boundary_transcript(c, 5)
    body = replay.body_for(projected, {"node_id": "n", "video_id": "v", "clip_id": 0}, boundary=True)
    doc, boundary = replay.restore_document(source(body), c, 5)
    assert boundary
    assert doc["source_end_sec"] == 5
    value = doc["passages"][0]["value"]
    assert value["text"] == "Safe early words."
    assert "speaker_id" not in value
    assert "Future" not in replay.dumps(doc)


def test_changed_content_is_rejected_instead_of_claiming_exact_replay():
    c = content()
    body = replay.body_for(c, {"node_id": "n", "video_id": "v", "clip_id": 0})
    s = source(body)
    c["description"] = "Changed memory."
    assert replay.restore_document(s, c, 10) == (None, False)


def test_cross_video_source_pointer_is_rejected():
    c = content()
    body = replay.body_for(c, {"node_id": "n", "video_id": "other", "clip_id": 0})
    wrong = source(body)
    wrong["video_id"] = "other"
    assert replay.restore_document(wrong, c, 10) == (None, False)
