from autocut.quality import assess_sentence_index
from autocut.semantic import SentenceIndex


def test_assess_sentence_index_recommends_ocr_for_bad_asr() -> None:
    sentence_index = SentenceIndex.model_validate(
        {
            "sentences": [
                {
                    "id": "S0001",
                    "source_segment_id": "seg_001",
                    "start": 0,
                    "end": 95,
                    "text": "嗯嗯啊啊啊啊这这这这这这这。",
                }
            ]
        }
    )

    report = assess_sentence_index(sentence_index)

    assert report.recommendation == "prefer_subtitle_or_ocr"
    assert report.score < 0.7
    assert "sentence_timestamps_have_large_spans" in report.reasons


def test_assess_sentence_index_allows_clean_asr() -> None:
    sentence_index = SentenceIndex.model_validate(
        {
            "sentences": [
                {
                    "id": "S0001",
                    "source_segment_id": "seg_001",
                    "start": 0,
                    "end": 4,
                    "text": "今天我们介绍自动化剪辑的整体流程。",
                },
                {
                    "id": "S0002",
                    "source_segment_id": "seg_002",
                    "start": 5,
                    "end": 9,
                    "text": "第一步是提取音频并生成字幕。",
                },
            ]
        }
    )

    report = assess_sentence_index(sentence_index)

    assert report.recommendation == "asr_usable"
    assert report.score > 0.8
