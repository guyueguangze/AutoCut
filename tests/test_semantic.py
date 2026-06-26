from autocut.semantic import (
    EditPlan,
    build_sentence_index,
    build_topic_blocks,
    compact_blocks_to_markdown,
    edit_plan_to_timeline,
    sentence_index_to_srt,
)
from autocut.transcript import Transcript


def test_build_sentence_index_from_transcript() -> None:
    transcript = Transcript.model_validate(
        {
            "source": "input.mp4",
            "segments": [
                {"id": "seg_001", "start": 0, "end": 1, "text": "第一句。"},
                {"id": "seg_002", "start": 1.5, "end": 3, "text": "第二句。"},
            ],
        }
    )

    index = build_sentence_index(transcript)

    assert index.source == "input.mp4"
    assert [sentence.id for sentence in index.sentences] == ["S0001", "S0002"]
    assert index.sentences[0].pause_after == 0.5


def test_build_sentence_index_splits_large_segment_with_char_timestamps() -> None:
    transcript = Transcript.model_validate(
        {
            "segments": [
                {
                    "id": "seg_001",
                    "start": 0,
                    "end": 4,
                    "text": "第一句。第二句。",
                    "chars": [
                        {"text": "第", "start": 0.0, "end": 0.2},
                        {"text": "一", "start": 0.2, "end": 0.4},
                        {"text": "句", "start": 0.4, "end": 0.6},
                        {"text": "。", "start": 0.6, "end": 0.8},
                        {"text": "第", "start": 2.0, "end": 2.2},
                        {"text": "二", "start": 2.2, "end": 2.4},
                        {"text": "句", "start": 2.4, "end": 2.6},
                        {"text": "。", "start": 2.6, "end": 2.8},
                    ],
                }
            ]
        }
    )

    index = build_sentence_index(transcript)

    assert [sentence.text for sentence in index.sentences] == ["第一句。", "第二句。"]
    assert index.sentences[0].start == 0
    assert index.sentences[0].end == 0.8
    assert index.sentences[0].pause_after == 1.2


def test_sentence_split_ignores_punctuation_when_mapping_char_timestamps() -> None:
    transcript = Transcript.model_validate(
        {
            "segments": [
                {
                    "id": "seg_001",
                    "start": 0,
                    "end": 4,
                    "text": "你好，世界。继续。",
                    "chars": [
                        {"text": "你", "start": 0.0, "end": 0.2},
                        {"text": "好", "start": 0.2, "end": 0.4},
                        {"text": "世", "start": 0.4, "end": 0.6},
                        {"text": "界", "start": 0.6, "end": 0.8},
                        {"text": "继", "start": 2.0, "end": 2.2},
                        {"text": "续", "start": 2.2, "end": 2.4},
                    ],
                }
            ]
        }
    )

    index = build_sentence_index(transcript)

    assert [sentence.text for sentence in index.sentences] == ["你好，世界。", "继续。"]
    assert index.sentences[0].end == 0.8
    assert index.sentences[1].start == 2.0


def test_sentence_split_uses_punctuation_timestamp_when_available() -> None:
    transcript = Transcript.model_validate(
        {
            "segments": [
                {
                    "id": "seg_001",
                    "start": 0,
                    "end": 4,
                    "text": "你好，世界。",
                    "chars": [
                        {"text": "你", "start": 0.0, "end": 0.2},
                        {"text": "好", "start": 0.2, "end": 0.4},
                        {"text": "，", "start": 0.4, "end": 0.5},
                        {"text": "世", "start": 0.5, "end": 0.7},
                        {"text": "界", "start": 0.7, "end": 0.9},
                        {"text": "。", "start": 0.9, "end": 1.0},
                    ],
                }
            ]
        }
    )

    index = build_sentence_index(transcript)

    assert index.sentences[0].start == 0.0
    assert index.sentences[0].end == 1.0


def test_build_topic_blocks_splits_on_max_duration() -> None:
    transcript = Transcript.model_validate(
        {
            "segments": [
                {"id": "seg_001", "start": 0, "end": 10, "text": "第一句。"},
                {"id": "seg_002", "start": 10, "end": 20, "text": "继续说明。"},
                {"id": "seg_003", "start": 20, "end": 40, "text": "补充案例。"},
            ]
        }
    )
    index = build_sentence_index(transcript)

    blocks = build_topic_blocks(index, min_duration=5, max_duration=25)

    assert [block.id for block in blocks.blocks] == ["B0001", "B0002"]
    assert blocks.blocks[0].start_sentence_id == "S0001"
    assert blocks.blocks[0].end_sentence_id == "S0002"


def test_compact_blocks_to_markdown_contains_block_ids() -> None:
    transcript = Transcript.model_validate(
        {"segments": [{"id": "seg_001", "start": 1, "end": 2, "text": "内容。"}]}
    )
    blocks = build_topic_blocks(build_sentence_index(transcript))

    markdown = compact_blocks_to_markdown(blocks)

    assert "[B0001 | 00:00:01.00-00:00:02.00 | 1 sentences]" in markdown
    assert "内容。" in markdown


def test_sentence_index_to_srt() -> None:
    transcript = Transcript.model_validate(
        {"segments": [{"id": "seg_001", "start": 1, "end": 2, "text": "内容。"}]}
    )
    index = build_sentence_index(transcript)

    assert sentence_index_to_srt(index) == "1\n00:00:01,000 --> 00:00:02,000\n内容。\n"


def test_edit_plan_to_timeline_uses_block_times() -> None:
    transcript = Transcript.model_validate(
        {
            "segments": [
                {"id": "seg_001", "start": 1, "end": 2, "text": "保留。"},
                {"id": "seg_002", "start": 3, "end": 4, "text": "删除。"},
            ]
        }
    )
    blocks = build_topic_blocks(build_sentence_index(transcript), max_duration=1.5)
    plan = EditPlan.model_validate(
        {
            "edit_decisions": [
                {"block_id": "B0001", "decision": "keep", "reason": "核心内容"},
                {"block_id": "B0002", "decision": "delete", "reason": "重复"},
            ]
        }
    )

    timeline = edit_plan_to_timeline(plan, blocks)

    assert [(item.id, item.start, item.end, item.decision) for item in timeline.segments] == [
        ("B0001", 1, 2, "keep"),
        ("B0002", 3, 4, "delete"),
    ]
