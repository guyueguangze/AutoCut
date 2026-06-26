import pytest

from autocut.timeline import Timeline


def test_keep_segments_are_sorted_and_filter_delete() -> None:
    timeline = Timeline.model_validate(
        {
            "segments": [
                {"id": "seg_002", "start": 10, "end": 12, "decision": "keep"},
                {"id": "seg_001", "start": 1, "end": 2, "decision": "delete"},
                {"id": "seg_003", "start": 5, "end": 6, "decision": "keep"},
            ]
        }
    )

    assert [segment.id for segment in timeline.keep_segments()] == ["seg_003", "seg_002"]


def test_segment_end_must_be_after_start() -> None:
    with pytest.raises(ValueError):
        Timeline.model_validate(
            {"segments": [{"id": "seg_001", "start": 2, "end": 2, "decision": "keep"}]}
        )
