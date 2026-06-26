from autocut.asr import funasr_result_to_transcript


def test_normalizes_funasr_character_timestamps() -> None:
    transcript = funasr_result_to_transcript(
        [
            {
                "text": "你好",
                "timestamp": [[0, 240], [240, 500]],
            }
        ],
        source="input.mp4",
    )

    assert transcript.source == "input.mp4"
    assert transcript.segments[0].text == "你好"
    assert transcript.segments[0].start == 0
    assert transcript.segments[0].end == 0.5
    assert [char.text for char in transcript.segments[0].chars] == ["你", "好"]


def test_normalizes_funasr_sentence_info() -> None:
    transcript = funasr_result_to_transcript(
        [
            {
                "sentence_info": [
                    {"text": "第一句。", "start": 0, "end": 1200, "spk": 0},
                    {"text": "第二句。", "start": 1500, "end": 2600, "spk": 1},
                ]
            }
        ]
    )

    assert [segment.text for segment in transcript.segments] == ["第一句。", "第二句。"]
    assert transcript.segments[0].speaker == "0"
    assert transcript.segments[1].start == 1.5
