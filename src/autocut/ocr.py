from pathlib import Path
from multiprocessing import get_context
from typing import Any

from pydantic import BaseModel, Field

from autocut.transcript import Transcript, TranscriptSegment


class OCRTextBox(BaseModel):
    text: str
    score: float = Field(ge=0, le=1)
    box: list[list[float]]


class OCRFrameResult(BaseModel):
    timestamp: float = Field(ge=0)
    image: str
    texts: list[OCRTextBox]


class OCRSample(BaseModel):
    timestamp: float = Field(ge=0)
    text: str = ""
    score: float = Field(default=0, ge=0, le=1)


class OCRSubtitleSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str
    score: float = Field(ge=0, le=1)


class OCRTimeRange(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)


OCRTask = tuple[int, str, float, float]
OCRArrayTask = tuple[int, bytes, tuple[int, int, int], float, float, str]

_OCR_WORKER_ENGINE: Any | None = None
_OCR_WORKER_ENGINE_NAME = "rapidocr"
_OCR_WORKER_MODEL_SIZE = "tiny"
_OCR_WORKER_DEVICE: str | None = None
_OCR_WORKER_INFERENCE_ENGINE: str | None = None

OCR_ENGINES = {"rapidocr", "ppocrv6"}
PPOCRV6_MODEL_SIZES = {"tiny", "small", "medium"}
MAX_SUBTITLE_CROP_HEIGHT = 96
ENDING_CREDIT_KEYWORDS = (
    "演员表",
    "片尾曲",
    "主题歌",
    "插曲",
    "作词",
    "作曲",
    "编曲",
    "演唱",
    "导演",
    "编剧",
    "主演",
    "领衔主演",
    "配音",
    "摄影",
    "摄像",
    "美术",
    "剪辑",
    "录音",
    "制片",
    "监制",
    "出品",
    "联合出品",
    "字幕",
    "鸣谢",
)
ENDING_SONG_PHRASES = (
    "你挑着担我牵着马",
    "迎来日出送走晚霞",
    "踏平坎坷成大道",
    "斗罢艰险又出发",
    "一番番春秋冬夏",
    "一场场酸甜苦辣",
    "敢问路在何方",
    "路在脚下",
)


def ocr_image(
    image: Path,
    *,
    timestamp: float = 0,
    engine_name: str = "rapidocr",
    model_size: str = "tiny",
    device: str | None = None,
    inference_engine: str | None = None,
) -> OCRFrameResult:
    if not image.exists():
        raise ValueError(f"Image file does not exist: {image}")

    engine = create_ocr_engine(
        engine_name=engine_name,
        model_size=model_size,
        device=device,
        inference_engine=inference_engine,
    )
    if _normalize_ocr_engine(engine_name) == "ppocrv6":
        return ocr_subtitle_image_rec_only(engine, image, timestamp=timestamp)
    return ocr_image_with_engine(engine, image, timestamp=timestamp)


def run_ocr_tasks(
    tasks: list[OCRTask],
    *,
    workers: int,
    ocr_threads: int | None = None,
    engine_name: str = "rapidocr",
    model_size: str = "tiny",
    device: str | None = None,
    inference_engine: str | None = None,
) -> list[tuple[int, OCRSample]]:
    if not tasks:
        return []
    if workers <= 1:
        _init_ocr_worker(
            ocr_threads,
            engine_name=engine_name,
            model_size=model_size,
            device=device,
            inference_engine=inference_engine,
        )
        return [_ocr_task_worker(task) for task in tasks]

    context = get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_ocr_worker,
        initargs=(
            ocr_threads if ocr_threads is not None else 1,
            engine_name,
            model_size,
            device,
            inference_engine,
        ),
    ) as pool:
        return list(pool.map(_ocr_task_worker, tasks))


def run_ocr_array_tasks(
    tasks: list[OCRArrayTask],
    *,
    workers: int,
    ocr_threads: int | None = None,
    engine_name: str = "rapidocr",
    model_size: str = "tiny",
    device: str | None = None,
    inference_engine: str | None = None,
) -> list[tuple[int, OCRSample]]:
    if not tasks:
        return []
    if workers <= 1:
        _init_ocr_worker(
            ocr_threads,
            engine_name=engine_name,
            model_size=model_size,
            device=device,
            inference_engine=inference_engine,
        )
        return [_ocr_array_task_worker(task) for task in tasks]

    context = get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_ocr_worker,
        initargs=(
            ocr_threads if ocr_threads is not None else 1,
            engine_name,
            model_size,
            device,
            inference_engine,
        ),
    ) as pool:
        return list(pool.map(_ocr_array_task_worker, tasks))


def run_ocr_array_tasks_batched(
    tasks: list[OCRArrayTask],
    *,
    workers: int,
    ocr_threads: int | None = None,
    engine_name: str = "rapidocr",
    model_size: str = "tiny",
    device: str | None = None,
    inference_engine: str | None = None,
    batch_size: int = 16,
) -> list[tuple[int, OCRSample]]:
    engine_name = _normalize_ocr_engine(engine_name)
    if engine_name != "ppocrv6":
        return run_ocr_array_tasks(
            tasks,
            workers=workers,
            ocr_threads=ocr_threads,
            engine_name=engine_name,
            model_size=model_size,
            device=device,
            inference_engine=inference_engine,
        )

    if not tasks:
        return []
    if batch_size <= 0:
        raise ValueError("OCR batch size must be greater than 0.")
    if batch_size == 1:
        _init_ocr_worker(
            ocr_threads,
            engine_name=engine_name,
            model_size=model_size,
            device=device,
            inference_engine=inference_engine,
        )
        return [_ocr_array_task_worker(task) for task in tasks]

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for in-memory OCR.") from exc

    engine = create_ocr_engine(
        engine_name=engine_name,
        model_size=model_size,
        device=device,
        inference_engine=inference_engine,
        ocr_threads=ocr_threads,
    )
    results: list[tuple[int, OCRSample]] = []
    crop_batch: list[object] = []
    meta_batch: list[tuple[int, float, float]] = []

    def flush_crop_batch() -> None:
        if not crop_batch:
            return
        raw_items = _ppocrv6_recognition_items(
            _ppocrv6_predict(engine._text_rec(), crop_batch, batch_size=batch_size)
        )
        for (index, timestamp, min_score), item in zip(meta_batch, raw_items, strict=False):
            text, score = item
            text = _normalize_ocr_text(text)
            if score >= min_score and _looks_like_subtitle(text):
                results.append(
                    (index, OCRSample(timestamp=timestamp, text=text, score=round(score, 4)))
                )
            else:
                results.append((index, OCRSample(timestamp=timestamp)))
        for index, timestamp, _ in meta_batch[len(raw_items) :]:
            results.append((index, OCRSample(timestamp=timestamp)))
        crop_batch.clear()
        meta_batch.clear()

    for task in tasks:
        index, frame_bytes, shape, timestamp, min_score, mode = task
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape)
        if mode == "rec_crop":
            crop_batch.append(frame_array)
            meta_batch.append((index, timestamp, min_score))
            if len(crop_batch) >= batch_size:
                flush_crop_batch()
            continue

        flush_crop_batch()
        if mode == "rec_frame":
            sample = ocr_subtitle_rec_only(
                engine,
                frame_array,
                timestamp=timestamp,
                min_score=min_score,
            )
            results.append((index, sample or OCRSample(timestamp=timestamp)))
            continue

        frame = ocr_array_with_engine(engine, frame_array, timestamp=timestamp)
        results.append((index, best_subtitle_text(frame, min_score=min_score)))

    flush_crop_batch()
    return sorted(results, key=lambda item: item[0])


def _init_ocr_worker(
    ocr_threads: int | None = None,
    engine_name: str = "rapidocr",
    model_size: str = "tiny",
    device: str | None = None,
    inference_engine: str | None = None,
) -> None:
    global _OCR_WORKER_ENGINE
    global _OCR_WORKER_ENGINE_NAME
    global _OCR_WORKER_MODEL_SIZE
    global _OCR_WORKER_DEVICE
    global _OCR_WORKER_INFERENCE_ENGINE
    requested_engine = _normalize_ocr_engine(engine_name)
    requested_model_size = _normalize_ppocrv6_model_size(model_size)
    if (
        _OCR_WORKER_ENGINE is not None
        and _OCR_WORKER_ENGINE_NAME == requested_engine
        and _OCR_WORKER_MODEL_SIZE == requested_model_size
        and _OCR_WORKER_DEVICE == device
        and _OCR_WORKER_INFERENCE_ENGINE == inference_engine
    ):
        return
    _OCR_WORKER_ENGINE_NAME = requested_engine
    _OCR_WORKER_MODEL_SIZE = requested_model_size
    _OCR_WORKER_DEVICE = device
    _OCR_WORKER_INFERENCE_ENGINE = inference_engine

    _OCR_WORKER_ENGINE = create_ocr_engine(
        engine_name=_OCR_WORKER_ENGINE_NAME,
        model_size=_OCR_WORKER_MODEL_SIZE,
        device=_OCR_WORKER_DEVICE,
        inference_engine=_OCR_WORKER_INFERENCE_ENGINE,
        ocr_threads=ocr_threads,
    )


def create_ocr_engine(
    *,
    engine_name: str = "rapidocr",
    model_size: str = "tiny",
    device: str | None = None,
    inference_engine: str | None = None,
    ocr_threads: int | None = None,
) -> object:
    engine_name = _normalize_ocr_engine(engine_name)
    if engine_name == "rapidocr":
        return _create_rapidocr_engine(ocr_threads=ocr_threads)

    model_size = _normalize_ppocrv6_model_size(model_size)
    return PaddleOCRV6Engine(
        model_size=model_size,
        device=device,
        inference_engine=inference_engine,
        cpu_threads=ocr_threads,
    )


def ensure_ocr_engine_available(engine_name: str) -> None:
    engine_name = _normalize_ocr_engine(engine_name)
    if engine_name == "rapidocr":
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "OCR dependencies are not installed. Run: python -m uv sync --extra ocr"
            ) from exc
        return

    try:
        import paddleocr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PP-OCRv6 dependencies are not installed. Run: python -m uv sync --extra ppocrv6"
        ) from exc


class PaddleOCRV6Engine:
    def __init__(
        self,
        *,
        model_size: str = "tiny",
        device: str | None = None,
        inference_engine: str | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        try:
            from paddleocr import PaddleOCR, TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "PP-OCRv6 dependencies are not installed. Run: python -m uv sync --extra ppocrv6"
            ) from exc

        model_size = _normalize_ppocrv6_model_size(model_size)
        self._paddle_ocr_cls = PaddleOCR
        self._text_recognition_cls = TextRecognition
        self._model_size = model_size
        self._common_kwargs = _ppocrv6_common_kwargs(
            device=device,
            inference_engine=inference_engine,
            cpu_threads=cpu_threads,
        )
        self.text_rec: object | None = None
        self.ocr: object | None = None

    def __call__(
        self,
        image: object,
        *,
        use_det: bool = True,
        use_cls: bool = True,
        use_rec: bool = True,
    ) -> tuple[list[Any], dict[str, Any]]:
        if use_rec and not use_det:
            raw_items = _ppocrv6_recognition_items(
                _ppocrv6_predict(self._text_rec(), image, batch_size=1)
            )
            return raw_items, {}

        raw_items = _ppocrv6_ocr_items(_ppocrv6_predict(self._ocr(), image))
        return raw_items, {}

    def _text_rec(self) -> object:
        if self.text_rec is None:
            self.text_rec = self._text_recognition_cls(
                model_name=f"PP-OCRv6_{self._model_size}_rec",
                **self._common_kwargs,
            )
        return self.text_rec

    def _ocr(self) -> object:
        if self.ocr is None:
            self.ocr = self._paddle_ocr_cls(
                text_detection_model_name=f"PP-OCRv6_{self._model_size}_det",
                text_recognition_model_name=f"PP-OCRv6_{self._model_size}_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                **self._common_kwargs,
            )
        return self.ocr


def _create_rapidocr_engine(*, ocr_threads: int | None = None) -> object:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are not installed. Run: python -m uv sync --extra ocr"
        ) from exc
    if ocr_threads is None:
        return RapidOCR()
    return RapidOCR(
        intra_op_num_threads=ocr_threads,
        inter_op_num_threads=1,
    )


def _normalize_ocr_engine(engine_name: str) -> str:
    value = engine_name.lower().strip()
    if value not in OCR_ENGINES:
        choices = ", ".join(sorted(OCR_ENGINES))
        raise ValueError(f"OCR engine must be one of: {choices}")
    return value


def _normalize_ppocrv6_model_size(model_size: str) -> str:
    value = model_size.lower().strip()
    if value not in PPOCRV6_MODEL_SIZES:
        choices = ", ".join(sorted(PPOCRV6_MODEL_SIZES))
        raise ValueError(f"PP-OCRv6 model size must be one of: {choices}")
    return value


def _ppocrv6_common_kwargs(
    *,
    device: str | None,
    inference_engine: str | None,
    cpu_threads: int | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if device:
        kwargs["device"] = device
    if inference_engine:
        kwargs["engine"] = inference_engine
    if cpu_threads is not None:
        kwargs["cpu_threads"] = cpu_threads
    return kwargs


def _ppocrv6_predict(model: object, image: object, **kwargs: object) -> object:
    predict = getattr(model, "predict")
    try:
        return predict(input=image, **kwargs)
    except TypeError:
        return predict(image, **kwargs)


def _ppocrv6_recognition_items(raw_result: object) -> list[tuple[str, float]]:
    items = []
    for result in _iter_result_items(raw_result):
        data = _result_data(result)
        text = data.get("rec_text") or data.get("text") or ""
        score = data.get("rec_score") or data.get("score") or 0.0
        if text:
            items.append((str(text), float(score)))
    return items


def _ppocrv6_ocr_items(raw_result: object) -> list[tuple[list[list[float]], str, float]]:
    items = []
    for result in _iter_result_items(raw_result):
        data = _result_data(result)
        texts = _as_list(data.get("rec_texts") or data.get("rec_text") or data.get("texts"))
        scores = _as_list(data.get("rec_scores") or data.get("rec_score") or data.get("scores"))
        boxes = _as_list(
            data.get("rec_polys")
            or data.get("dt_polys")
            or data.get("rec_boxes")
            or data.get("boxes")
        )
        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else 0.0
            box = boxes[index] if index < len(boxes) else []
            items.append((_box_to_points(box), str(text), float(score)))
    return items


def _iter_result_items(raw_result: object) -> list[object]:
    if raw_result is None:
        return []
    if isinstance(raw_result, (str, bytes, dict)):
        return [raw_result]
    try:
        return list(raw_result)
    except TypeError:
        return [raw_result]


def _result_data(result: object) -> dict[str, object]:
    if isinstance(result, dict):
        data: object = result
    else:
        data = getattr(result, "json", None)
        if callable(data):
            data = data()
        if data is None:
            try:
                data = dict(result)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                data = {}
    if isinstance(data, dict) and isinstance(data.get("res"), dict):
        data = data["res"]
    return data if isinstance(data, dict) else {}


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        return [value]


def _box_to_points(box: object) -> list[list[float]]:
    if hasattr(box, "tolist"):
        box = box.tolist()
    if isinstance(box, (list, tuple)) and len(box) == 4 and all(
        isinstance(value, (int, float)) for value in box
    ):
        x1, y1, x2, y2 = (float(value) for value in box)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if isinstance(box, (list, tuple)):
        points = []
        for point in box:
            if hasattr(point, "tolist"):
                point = point.tolist()
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append([float(point[0]), float(point[1])])
        if points:
            return points
    return []


def _bbox_to_points(bbox: tuple[int, int, int, int] | None) -> list[list[float]]:
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    return [
        [float(x1), float(y1)],
        [float(x2), float(y1)],
        [float(x2), float(y2)],
        [float(x1), float(y2)],
    ]


def _ocr_task_worker(task: OCRTask) -> tuple[int, OCRSample]:
    index, image, timestamp, min_score = task
    if _OCR_WORKER_ENGINE is None:
        _init_ocr_worker()
    frame = ocr_image_with_engine(_OCR_WORKER_ENGINE, Path(image), timestamp=timestamp)
    return index, best_subtitle_text(frame, min_score=min_score)


def _ocr_array_task_worker(task: OCRArrayTask) -> tuple[int, OCRSample]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for in-memory OCR.") from exc

    index, frame_bytes, shape, timestamp, min_score, mode = task
    if _OCR_WORKER_ENGINE is None:
        _init_ocr_worker()
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape)
    if mode == "rec_crop":
        sample = ocr_text_crop_rec_only(
            _OCR_WORKER_ENGINE,
            frame_array,
            timestamp=timestamp,
            min_score=min_score,
        )
        if sample is not None:
            return index, sample
        return index, OCRSample(timestamp=timestamp)

    if mode == "rec_frame":
        sample = ocr_subtitle_rec_only(
            _OCR_WORKER_ENGINE,
            frame_array,
            timestamp=timestamp,
            min_score=min_score,
        )
        if sample is not None:
            return index, sample
        if _OCR_WORKER_ENGINE_NAME == "ppocrv6":
            return index, OCRSample(timestamp=timestamp)

    frame = ocr_array_with_engine(_OCR_WORKER_ENGINE, frame_array, timestamp=timestamp)
    return index, best_subtitle_text(frame, min_score=min_score)


def ocr_image_with_engine(engine: object, image: Path, *, timestamp: float = 0) -> OCRFrameResult:
    if not image.exists():
        raise ValueError(f"Image file does not exist: {image}")

    raw_result, _ = engine(str(image))
    texts = []
    for item in raw_result or []:
        box, text, score = item
        texts.append(OCRTextBox(text=text, score=round(float(score), 4), box=box))
    return OCRFrameResult(timestamp=timestamp, image=str(image), texts=texts)


def ocr_array_with_engine(engine: object, image: object, *, timestamp: float = 0) -> OCRFrameResult:
    raw_result, _ = engine(image)
    texts = []
    for item in raw_result or []:
        box, text, score = item
        texts.append(OCRTextBox(text=text, score=round(float(score), 4), box=box))
    return OCRFrameResult(timestamp=timestamp, image="<memory>", texts=texts)


def ocr_subtitle_image_rec_only(
    engine: object,
    image: Path,
    *,
    timestamp: float = 0,
    min_score: float = 0.0,
) -> OCRFrameResult:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: python -m uv sync --extra ocr"
        ) from exc

    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError(f"Could not read image: {image}")

    sample = ocr_subtitle_rec_only(engine, frame, timestamp=timestamp, min_score=min_score)
    texts = []
    if sample is not None and sample.text:
        bbox = find_subtitle_bbox(frame)
        box = _bbox_to_points(bbox) if bbox is not None else []
        texts.append(OCRTextBox(text=sample.text, score=sample.score, box=box))
    return OCRFrameResult(timestamp=timestamp, image=str(image), texts=texts)


def ocr_text_crop_rec_only(
    engine: object,
    crop: object,
    *,
    timestamp: float = 0,
    min_score: float = 0.7,
) -> OCRSample | None:
    if crop.shape[0] <= 0 or crop.shape[1] <= 0:
        return None
    raw_result, _ = engine(crop, use_det=False, use_cls=False, use_rec=True)
    if not raw_result:
        return None

    best = max(raw_result, key=lambda item: float(item[1]) if len(item) > 1 else 0.0)
    text = _normalize_ocr_text(str(best[0]))
    score = float(best[1]) if len(best) > 1 else 0.0
    if score < min_score or not _looks_like_subtitle(text):
        return None
    return OCRSample(timestamp=timestamp, text=text, score=round(score, 4))


def ocr_subtitle_rec_only(
    engine: object,
    image: object,
    *,
    timestamp: float = 0,
    min_score: float = 0.7,
) -> OCRSample | None:
    bbox = find_subtitle_bbox(image)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] <= 0 or crop.shape[1] <= 0:
        return None
    if crop.shape[0] > MAX_SUBTITLE_CROP_HEIGHT:
        return None

    return ocr_text_crop_rec_only(engine, crop, timestamp=timestamp, min_score=min_score)


def best_subtitle_text(
    frame: OCRFrameResult,
    *,
    min_score: float = 0.7,
) -> OCRSample:
    candidates = [
        item
        for item in frame.texts
        if item.score >= min_score and _looks_like_subtitle(item.text)
    ]
    if not candidates:
        return OCRSample(timestamp=frame.timestamp)
    best = max(candidates, key=lambda item: (item.score, len(item.text)))
    return OCRSample(timestamp=frame.timestamp, text=_normalize_ocr_text(best.text), score=best.score)


def merge_ocr_samples(
    samples: list[OCRSample],
    *,
    interval: float,
    min_duration: float = 0.4,
    max_gap: float | None = None,
) -> list[OCRSubtitleSegment]:
    if not samples:
        return []

    gap_limit = max_gap if max_gap is not None else interval * 1.5
    segments: list[OCRSubtitleSegment] = []
    active_samples: list[OCRSample] = []
    active_end = 0.0

    def flush() -> None:
        nonlocal active_samples, active_end
        if active_samples:
            active_start = active_samples[0].timestamp
            text = _voted_subtitle_text(active_samples)
            scores = [sample.score for sample in active_samples]
        else:
            active_start = 0.0
            text = ""
            scores = []
        if text and active_end - active_start >= min_duration:
            segments.append(
                OCRSubtitleSegment(
                    start=round(active_start, 3),
                    end=round(active_end, 3),
                    text=text,
                    score=round(sum(scores) / len(scores), 4),
                )
            )
        active_samples = []

    for sample in samples:
        text = _normalize_ocr_text(sample.text)
        if not text:
            flush()
            continue

        sample_end = sample.timestamp + interval
        normalized_sample = OCRSample(
            timestamp=sample.timestamp,
            text=text,
            score=sample.score,
        )
        if (
            active_samples
            and sample.timestamp - active_end <= gap_limit
            and _belongs_to_subtitle_event(active_samples, text)
        ):
            active_samples.append(normalized_sample)
            active_end = sample_end
            continue

        flush()
        active_samples = [normalized_sample]
        active_end = sample_end

    flush()
    return segments


def ocr_segments_to_transcript(
    segments: list[OCRSubtitleSegment],
    *,
    source: str | None = None,
) -> Transcript:
    return Transcript(
        source=source,
        segments=[
            TranscriptSegment(
                id=f"ocr_{index:04d}",
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
            for index, segment in enumerate(segments, start=1)
        ],
    )


def filter_trailing_credit_segments(
    segments: list[OCRSubtitleSegment],
    *,
    max_gap: float = 12.0,
    min_tail_segments: int = 4,
    min_tail_duration: float = 15.0,
) -> list[OCRSubtitleSegment]:
    if len(segments) < min_tail_segments:
        return segments

    start_index = _trailing_credit_start_index(
        segments,
        max_gap=max_gap,
        min_tail_segments=min_tail_segments,
        min_tail_duration=min_tail_duration,
    )
    if start_index is None:
        return segments
    return segments[:start_index]


def detect_subtitle_ranges(
    detections: list[tuple[float, bool]],
    *,
    interval: float,
    expand: float = 1.0,
    max_gap: float | None = None,
    start_bound: float | None = None,
    end_bound: float | None = None,
) -> list[OCRTimeRange]:
    gap_limit = max_gap if max_gap is not None else interval * 1.5
    raw_ranges: list[OCRTimeRange] = []
    active_start: float | None = None
    active_end: float | None = None

    for timestamp, has_subtitle in detections:
        if not has_subtitle:
            continue
        sample_end = timestamp + interval
        if active_start is None:
            active_start = timestamp
            active_end = sample_end
            continue
        if timestamp - (active_end or timestamp) <= gap_limit:
            active_end = sample_end
            continue
        raw_ranges.append(OCRTimeRange(start=active_start, end=active_end or sample_end))
        active_start = timestamp
        active_end = sample_end

    if active_start is not None:
        raw_ranges.append(OCRTimeRange(start=active_start, end=active_end or active_start + interval))

    expanded = []
    for item in raw_ranges:
        start = item.start - expand
        end = item.end + expand
        if start_bound is not None:
            start = max(start_bound, start)
        else:
            start = max(0, start)
        if end_bound is not None:
            end = min(end_bound, end)
        if end > start:
            expanded.append(OCRTimeRange(start=round(start, 3), end=round(end, 3)))

    return _merge_ranges(expanded)


def subtitle_frame_signature(
    image: Path,
    *,
    min_white_pixels: int = 120,
) -> bytes | None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: python -m uv sync --extra ocr"
        ) from exc

    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError(f"Could not read image: {image}")

    return subtitle_frame_signature_from_array(frame, min_white_pixels=min_white_pixels)


def subtitle_frame_signature_from_array(
    frame: object,
    *,
    min_white_pixels: int = 120,
) -> bytes | None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: python -m uv sync --extra ocr"
        ) from exc

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    white_mask = ((gray > 175) & (saturation < 90)).astype("uint8")
    if int(white_mask.sum()) < min_white_pixels:
        return None

    small = cv2.resize(white_mask * 255, (64, 16), interpolation=cv2.INTER_AREA)
    bits = (small > 24).astype(np.uint8).reshape(-1)
    return np.packbits(bits).tobytes()


def find_subtitle_bbox(
    frame: object,
    *,
    min_white_pixels: int = 80,
    padding_x: int = 2,
    padding_y: int = 2,
) -> tuple[int, int, int, int] | None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run: python -m uv sync --extra ocr"
        ) from exc

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = ((gray > 175) & (saturation < 90)).astype("uint8")

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(mask.shape, dtype="uint8")
    height, width = mask.shape
    for label in range(1, labels_count):
        x, y, w, h, area = stats[label]
        if area < 3 or h < 3:
            continue
        if y < height * 0.1:
            continue
        if x < width * 0.05 or x + w > width * 0.95:
            continue
        keep[labels == label] = 1

    if int(keep.sum()) < min_white_pixels:
        return None

    ys, xs = np.where(keep > 0)
    x1 = max(0, int(xs.min()) - padding_x)
    x2 = min(width, int(xs.max()) + padding_x + 1)
    y1 = max(0, int(ys.min()) - padding_y)
    y2 = min(height, int(ys.max()) + padding_y + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def signature_distance(left: bytes | None, right: bytes | None) -> int | None:
    if left is None or right is None:
        return None
    if len(left) != len(right):
        return None
    return sum((a ^ b).bit_count() for a, b in zip(left, right, strict=True))


def _merge_ranges(ranges: list[OCRTimeRange]) -> list[OCRTimeRange]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda item: item.start)
    merged = [sorted_ranges[0]]
    for item in sorted_ranges[1:]:
        current = merged[-1]
        if item.start <= current.end:
            current.end = max(current.end, item.end)
            continue
        merged.append(item)
    return merged


def _looks_like_subtitle(text: str) -> bool:
    normalized = _normalize_ocr_text(text)
    if len(normalized) < 2:
        return False
    lowered = normalized.lower()
    blocked = ("cctv", "bilibili", "bibi", "电视剧")
    return not any(token in lowered for token in blocked)


def _trailing_credit_start_index(
    segments: list[OCRSubtitleSegment],
    *,
    max_gap: float,
    min_tail_segments: int,
    min_tail_duration: float,
) -> int | None:
    index = len(segments) - 1
    while index > 0 and segments[index].start - segments[index - 1].end <= max_gap:
        index -= 1

    tail = segments[index:]
    if len(tail) < min_tail_segments:
        return None
    if _credit_signal(tail[0].text) == 0:
        return None
    if tail[-1].end - tail[0].start < min_tail_duration:
        return None
    if _is_credit_tail(tail):
        return index
    return None


def _is_credit_tail(segments: list[OCRSubtitleSegment]) -> bool:
    signals = [_credit_signal(segment.text) for segment in segments]
    signal_count = sum(1 for signal in signals if signal > 0)
    strong_count = sum(1 for signal in signals if signal >= 2)
    if strong_count >= 2 and signal_count >= max(3, len(segments) // 3):
        return True
    if len(segments) >= 6 and _repeated_song_tail(segments):
        return True
    return False


def _credit_signal(text: str) -> int:
    normalized = _normalize_ocr_text(text)
    if not normalized:
        return 0
    score = 0
    if any(keyword in normalized for keyword in ENDING_CREDIT_KEYWORDS):
        score += 2
    song_hits = sum(1 for phrase in ENDING_SONG_PHRASES if phrase in normalized)
    score += song_hits
    if "啦" in normalized and len(normalized) <= 6:
        score += 1
    return score


def _repeated_song_tail(segments: list[OCRSubtitleSegment]) -> bool:
    hits = [
        segment
        for segment in segments
        if any(phrase in _normalize_ocr_text(segment.text) for phrase in ENDING_SONG_PHRASES)
    ]
    if len(hits) < 4:
        return False
    return hits[-1].end - hits[0].start >= 12


def _belongs_to_subtitle_event(active_samples: list[OCRSample], text: str) -> bool:
    active_texts = [sample.text for sample in active_samples if sample.text]
    if not active_texts:
        return False
    if text in active_texts:
        return True

    stable_count = _dominant_text_count(active_texts)
    references = list(dict.fromkeys([active_texts[-1], _voted_subtitle_text(active_samples)]))
    for reference in references:
        if _stable_repeated_subtitle_text(reference, text) is not None:
            if _is_short_cjk_substitution(reference, text):
                continue
            return True
        if stable_count < 2 and _is_multi_character_ocr_jitter(reference, text):
            return True
    return False


def _voted_subtitle_text(samples: list[OCRSample]) -> str:
    text_stats: dict[str, dict[str, float]] = {}
    for index, sample in enumerate(samples):
        text = sample.text
        if not text:
            continue
        stats = text_stats.setdefault(
            text,
            {"count": 0.0, "score": 0.0, "first_index": float(index), "last_index": float(index)},
        )
        stats["count"] += 1
        stats["score"] += sample.score
        stats["last_index"] = float(index)

    if not text_stats:
        return ""
    if len(text_stats) == 1:
        return next(iter(text_stats))

    if _is_single_character_variant_family(text_stats):
        return _first_observed_text(text_stats)

    exact_best = _best_exact_subtitle_text(text_stats)
    top_count = text_stats[exact_best]["count"]
    tied_text_stats = {
        text: stats for text, stats in text_stats.items() if stats["count"] == top_count
    }
    if len(tied_text_stats) == 1:
        return exact_best

    same_length = len({len(text) for text in text_stats}) == 1
    if same_length:
        return _character_supported_subtitle_text(samples, tied_text_stats)
    return exact_best


def _best_exact_subtitle_text(text_stats: dict[str, dict[str, float]]) -> str:
    return max(
        text_stats,
        key=lambda text: (
            text_stats[text]["count"],
            text_stats[text]["score"],
            -text_stats[text]["first_index"],
        ),
    )


def _first_observed_text(text_stats: dict[str, dict[str, float]]) -> str:
    return min(text_stats, key=lambda text: text_stats[text]["first_index"])


def _is_single_character_variant_family(text_stats: dict[str, dict[str, float]]) -> bool:
    texts = list(text_stats)
    if len({len(text) for text in texts}) != 1:
        return False
    first = _first_observed_text(text_stats)
    return all(text == first or _hamming_distance(first, text) == 1 for text in texts)


def _character_supported_subtitle_text(
    samples: list[OCRSample],
    text_stats: dict[str, dict[str, float]],
) -> str:
    char_stats: list[dict[str, dict[str, float]]] = [
        {} for _ in range(len(next(iter(text_stats))))
    ]
    for sample in samples:
        if not sample.text:
            continue
        for index, char in enumerate(sample.text):
            stats = char_stats[index].setdefault(char, {"count": 0.0, "score": 0.0})
            stats["count"] += 1
            stats["score"] += sample.score

    def support(text: str) -> tuple[float, float, float, float, float]:
        char_count = 0.0
        char_score = 0.0
        for index, char in enumerate(text):
            stats = char_stats[index][char]
            char_count += stats["count"]
            char_score += stats["score"]
        exact = text_stats[text]
        return (
            char_count,
            char_score,
            exact["count"],
            exact["score"],
            exact["last_index"],
        )

    return max(text_stats, key=support)


def _dominant_text_count(texts: list[str]) -> int:
    counts: dict[str, int] = {}
    for text in texts:
        counts[text] = counts.get(text, 0) + 1
    return max(counts.values(), default=0)


def _is_short_cjk_substitution(left: str, right: str) -> bool:
    if len(left) != len(right) or len(left) >= 4:
        return False
    if not all("\u4e00" <= char <= "\u9fff" for char in left + right):
        return False
    return _hamming_distance(left, right) == 1


def _is_multi_character_ocr_jitter(left: str, right: str) -> bool:
    if len(left) != len(right) or len(left) < 8:
        return False
    distance = _hamming_distance(left, right)
    return 1 < distance <= max(2, len(left) // 8)


def _stable_repeated_subtitle_text(left: str, right: str) -> str | None:
    if not left or not right:
        return None
    if left == right:
        return left

    if len(left) == len(right) and _hamming_distance(left, right) == 1:
        return left
    if len(left) + 1 == len(right):
        return left if _is_single_edge_noise(longer=right, shorter=left) else None
    if len(right) + 1 == len(left):
        return right if _is_single_edge_noise(longer=left, shorter=right) else None
    return None


def _hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length strings.")
    return sum(1 for left_char, right_char in zip(left, right, strict=True) if left_char != right_char)


def _is_single_edge_noise(*, longer: str, shorter: str) -> bool:
    if longer[1:] == shorter:
        noise = longer[0]
    elif longer[:-1] == shorter:
        noise = longer[-1]
    else:
        return False
    return noise.isascii() and noise.isalnum()


def _normalize_ocr_text(text: str) -> str:
    return "".join(text.split()).strip("。！？!?；;，,、：:,.…“”\"'（）()《》|· ")
