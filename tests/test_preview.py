from autocut.preview import transcript_to_preview_transcript
from autocut.timeline import Timeline
from autocut.transcript import Transcript


def test_transcript_to_preview_transcript_remaps_keep_segments() -> None:
    transcript = Transcript.model_validate(
        {
            "segments": [
                {"id": "seg_001", "start": 10, "end": 12, "text": "第一段"},
                {"id": "seg_002", "start": 20, "end": 22, "text": "第二段"},
            ]
        }
    )
    timeline = Timeline.model_validate(
        {
            "segments": [
                {"id": "B0001", "start": 10, "end": 13, "decision": "keep"},
                {"id": "B0002", "start": 20, "end": 23, "decision": "keep"},
            ]
        }
    )

    preview = transcript_to_preview_transcript(transcript, timeline)

    assert [(item.start, item.end, item.text) for item in preview.segments] == [
        (0, 2, "第一段"),
        (3, 5, "第二段"),
    ]


def test_transcript_to_preview_transcript_clips_overlapping_subtitles() -> None:
    transcript = Transcript.model_validate(
        {"segments": [{"id": "seg_001", "start": 9, "end": 12, "text": "跨段字幕"}]}
    )
    timeline = Timeline.model_validate(
        {"segments": [{"id": "B0001", "start": 10, "end": 11, "decision": "keep"}]}
    )

    preview = transcript_to_preview_transcript(transcript, timeline)

    assert [(item.start, item.end, item.text) for item in preview.segments] == [
        (0, 1, "跨段字幕")
    ]
