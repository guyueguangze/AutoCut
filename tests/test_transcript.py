from autocut.transcript import Transcript, format_srt_time, transcript_to_srt


def test_format_srt_time_rounds_to_milliseconds() -> None:
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(65.4321) == "00:01:05,432"
    assert format_srt_time(3661.9996) == "01:01:02,000"


def test_transcript_to_srt() -> None:
    transcript = Transcript.model_validate(
        {
            "segments": [
                {
                    "id": "seg_001",
                    "start": 0,
                    "end": 1.25,
                    "text": "第一句。",
                },
                {
                    "id": "seg_002",
                    "start": 1.5,
                    "end": 3,
                    "text": "第二句。",
                },
            ]
        }
    )

    assert transcript_to_srt(transcript) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "第一句。\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 00:00:03,000\n"
        "第二句。\n"
    )
