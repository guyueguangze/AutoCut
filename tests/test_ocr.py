from autocut.ocr import (
    OCRFrameResult,
    OCRSample,
    _normalize_ocr_engine,
    _normalize_ppocrv6_model_size,
    _ppocrv6_ocr_items,
    _ppocrv6_recognition_items,
    run_ocr_array_tasks_batched,
    best_subtitle_text,
    detect_subtitle_ranges,
    filter_trailing_credit_segments,
    merge_ocr_samples,
)


def test_ocr_frame_result_model() -> None:
    result = OCRFrameResult.model_validate(
        {
            "timestamp": 12.5,
            "image": "frame.png",
            "texts": [
                {
                    "text": "不中听不中听",
                    "score": 0.99,
                    "box": [[292, 524], [480, 524], [480, 559], [292, 559]],
                }
            ],
        }
    )

    assert result.texts[0].text == "不中听不中听"
    assert result.timestamp == 12.5


def test_best_subtitle_text_filters_logos() -> None:
    frame = OCRFrameResult.model_validate(
        {
            "timestamp": 1,
            "image": "frame.png",
            "texts": [
                {"text": "CCTV8", "score": 0.99, "box": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                {
                    "text": "不中听 不 中听",
                    "score": 0.95,
                    "box": [[0, 0], [1, 0], [1, 1], [0, 1]],
                },
            ],
        }
    )

    sample = best_subtitle_text(frame)

    assert sample.text == "不中听不中听"


def test_merge_ocr_samples_merges_repeated_text() -> None:
    segments = merge_ocr_samples(
        [
            OCRSample(timestamp=0, text="第一句", score=0.9),
            OCRSample(timestamp=1, text="第一句", score=0.8),
            OCRSample(timestamp=2, text="", score=0),
            OCRSample(timestamp=3, text="第二句", score=0.95),
        ],
        interval=1,
    )

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0, 2, "第一句"),
        (3, 4, "第二句"),
    ]


def test_merge_ocr_samples_suppresses_single_ascii_edge_noise() -> None:
    segments = merge_ocr_samples(
        [
            OCRSample(timestamp=0, text="下界是何声响", score=0.99),
            OCRSample(timestamp=0.5, text="C下界是何声响", score=0.9),
            OCRSample(timestamp=1, text="下界是何声响", score=0.99),
        ],
        interval=0.5,
    )

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0, 1.5, "下界是何声响"),
    ]


def test_merge_ocr_samples_suppresses_one_character_substitution_noise() -> None:
    segments = merge_ocr_samples(
        [
            OCRSample(timestamp=0, text="它来啦它来啦", score=0.99),
            OCRSample(timestamp=1, text="官来啦它来啦", score=0.9),
        ],
        interval=1,
    )

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0, 2, "它来啦它来啦"),
    ]


def test_merge_ocr_samples_keeps_first_text_for_single_character_variant_family() -> None:
    segments = merge_ocr_samples(
        [
            OCRSample(timestamp=0, text="我看你相貌举止像个糊狱", score=0.88),
            OCRSample(timestamp=1, text="我看你相貌举止像个糊狲", score=0.92),
            OCRSample(timestamp=2, text="我看你相貌举止像个糊狲", score=0.93),
        ],
        interval=1,
    )

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0, 3, "我看你相貌举止像个糊狱"),
    ]


def test_merge_ocr_samples_clusters_multi_character_ocr_jitter() -> None:
    segments = merge_ocr_samples(
        [
            OCRSample(timestamp=0, text="我看你相貌举止像个糊狱", score=0.88),
            OCRSample(timestamp=1, text="我看你相貌举止像个猢孙", score=0.9),
            OCRSample(timestamp=2, text="我看你相貌举止像个糊狲", score=0.93),
            OCRSample(timestamp=3, text="我看你相貌举止像个糊狲", score=0.93),
        ],
        interval=1,
    )

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0, 4, "我看你相貌举止像个糊狲"),
    ]


def test_merge_ocr_samples_keeps_stable_adjacent_short_subtitles_separate() -> None:
    segments = merge_ocr_samples(
        [
            OCRSample(timestamp=0, text="去吧", score=0.95),
            OCRSample(timestamp=1, text="去吧", score=0.95),
            OCRSample(timestamp=2, text="来吧", score=0.95),
            OCRSample(timestamp=3, text="来吧", score=0.95),
        ],
        interval=1,
    )

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0, 2, "去吧"),
        (2, 4, "来吧"),
    ]


def test_merge_ocr_samples_does_not_vote_across_blank_frame() -> None:
    segments = merge_ocr_samples(
        [
            OCRSample(timestamp=0, text="糊狲", score=0.95),
            OCRSample(timestamp=1, text="糊狲", score=0.95),
            OCRSample(timestamp=2, text="", score=0),
            OCRSample(timestamp=3, text="猢孙", score=0.95),
            OCRSample(timestamp=4, text="猢孙", score=0.95),
        ],
        interval=1,
    )

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0, 2, "糊狲"),
        (3, 5, "猢孙"),
    ]


def test_filter_trailing_credit_segments_removes_song_and_credit_tail() -> None:
    segments = [
        _ocr_segment(100, 103, "弟子记下了"),
        _ocr_segment(120, 128, "你挑着担我牵着马"),
        _ocr_segment(129, 132, "迎来日出送走晚霞"),
        _ocr_segment(132, 134, "玉皇大童来日出送走晚霞玉姜"),
        _ocr_segment(134, 136, "踏平坎坷成大道"),
        _ocr_segment(136, 138, "顺风耳踏平坎坷成大道项汉"),
        _ocr_segment(138, 141, "斗罢艰险又出发又出发"),
        _ocr_segment(141, 144, "啦…啦"),
        _ocr_segment(144, 148, "敢问路在何方路在脚下"),
    ]

    filtered = filter_trailing_credit_segments(segments)

    assert [(item.start, item.end, item.text) for item in filtered] == [
        (100, 103, "弟子记下了"),
    ]


def test_filter_trailing_credit_segments_keeps_inline_song_like_dialogue() -> None:
    segments = [
        _ocr_segment(100, 103, "你挑着担我牵着马"),
        _ocr_segment(103, 106, "迎来日出送走晚霞"),
        _ocr_segment(106, 109, "师父我们到了"),
        _ocr_segment(109, 112, "好好好"),
        _ocr_segment(112, 115, "先歇一歇"),
    ]

    assert filter_trailing_credit_segments(segments) == segments


def test_filter_trailing_credit_segments_only_considers_final_continuous_tail() -> None:
    segments = [
        _ocr_segment(100, 103, "弟子记下了"),
        _ocr_segment(110, 112, "啦…啦"),
        _ocr_segment(120, 122, "师父"),
        _ocr_segment(150, 153, "导演杨洁"),
        _ocr_segment(153, 156, "编剧戴英禄"),
        _ocr_segment(156, 159, "作曲许镜清"),
        _ocr_segment(159, 162, "演唱蒋大为"),
    ]

    filtered = filter_trailing_credit_segments(
        segments,
        min_tail_segments=3,
        min_tail_duration=6,
    )

    assert [(item.start, item.end, item.text) for item in filtered] == [
        (100, 103, "弟子记下了"),
        (110, 112, "啦…啦"),
        (120, 122, "师父"),
    ]


def test_filter_trailing_credit_segments_removes_keyword_credit_tail() -> None:
    segments = [
        _ocr_segment(100, 103, "师父弟子告退"),
        _ocr_segment(120, 122, "导演杨洁"),
        _ocr_segment(122, 124, "编剧戴英禄"),
        _ocr_segment(124, 126, "作曲许镜清"),
        _ocr_segment(126, 128, "演唱蒋大为"),
    ]

    filtered = filter_trailing_credit_segments(
        segments,
        min_tail_segments=3,
        min_tail_duration=6,
    )

    assert [(item.start, item.end, item.text) for item in filtered] == [
        (100, 103, "师父弟子告退"),
    ]


def test_detect_subtitle_ranges_expands_and_merges_hits() -> None:
    ranges = detect_subtitle_ranges(
        [
            (0, False),
            (1, True),
            (2, True),
            (6, True),
            (9, False),
        ],
        interval=1,
        expand=1,
        max_gap=1.5,
        start_bound=0,
        end_bound=10,
    )

    assert [(item.start, item.end) for item in ranges] == [(0, 4), (5, 8)]


def test_ocr_engine_validation_accepts_supported_values() -> None:
    assert _normalize_ocr_engine("PPOCRV6") == "ppocrv6"
    assert _normalize_ppocrv6_model_size("SMALL") == "small"


def test_ppocrv6_recognition_result_parser() -> None:
    raw = [{"res": {"rec_text": "第一句", "rec_score": 0.92}}]

    assert _ppocrv6_recognition_items(raw) == [("第一句", 0.92)]


def test_ppocrv6_ocr_result_parser() -> None:
    raw = [
        {
            "res": {
                "rec_texts": ["第一句"],
                "rec_scores": [0.91],
                "rec_boxes": [[10, 20, 110, 40]],
            }
        }
    ]

    assert _ppocrv6_ocr_items(raw) == [
        ([[10.0, 20.0], [110.0, 20.0], [110.0, 40.0], [10.0, 40.0]], "第一句", 0.91)
    ]


def test_batched_runner_delegates_non_ppocrv6_engine(monkeypatch) -> None:
    calls = {}

    def fake_run(tasks, **kwargs):
        calls["tasks"] = tasks
        calls["kwargs"] = kwargs
        return []

    monkeypatch.setattr("autocut.ocr.run_ocr_array_tasks", fake_run)

    task = (0, b"\x00\x00\x00", (1, 1, 3), 0.0, 0.7, "rec_crop")
    assert (
        run_ocr_array_tasks_batched(
            [task],
            workers=2,
            engine_name="rapidocr",
            batch_size=4,
        )
        == []
    )
    assert calls["kwargs"]["workers"] == 2


def _ocr_segment(start: float, end: float, text: str, score: float = 0.9):
    from autocut.ocr import OCRSubtitleSegment

    return OCRSubtitleSegment(start=start, end=end, text=text, score=score)
