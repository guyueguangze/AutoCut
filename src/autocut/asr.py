import os
from pathlib import Path

from autocut.ffmpeg import ffmpeg_bin_dir
from autocut.transcript import Transcript, TranscriptChar, TranscriptSegment


class ASRError(RuntimeError):
    """Raised when speech recognition cannot complete."""


def transcribe_placeholder(video: Path) -> str:
    return (
        "ASR module is not wired yet. Planned pipeline: extract audio with FFmpeg, "
        "run FunASR, then refine character timestamps with Qwen3-ForcedAligner. "
        f"Input: {video}"
    )


class FunASRAdapter:
    """Thin boundary for the future FunASR integration."""

    def __init__(
        self,
        model: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
        device: str = "cpu",
    ) -> None:
        self.model_name = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.device = device

    def transcribe(self, audio: Path, source: Path | None = None) -> Transcript:
        if not audio.exists():
            raise ASRError(f"Audio file does not exist: {audio}")

        _ensure_ffmpeg_on_path()

        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ASRError(
                "FunASR is not installed. Run: python -m uv sync --extra asr"
            ) from exc

        model = AutoModel(
            model=self.model_name,
            vad_model=self.vad_model,
            vad_kwargs={"max_single_segment_time": 60000},
            punc_model=self.punc_model,
            device=self.device,
            disable_update=True,
        )
        result = model.generate(input=str(audio), batch_size_s=300)
        return funasr_result_to_transcript(result, source=source or audio)


def funasr_result_to_transcript(result: object, source: Path | str | None = None) -> Transcript:
    items = result if isinstance(result, list) else [result]
    segments: list[TranscriptSegment] = []

    for item_index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        sentence_info = item.get("sentence_info")
        if isinstance(sentence_info, list) and sentence_info:
            segments.extend(_segments_from_sentence_info(sentence_info, offset=len(segments)))
            continue

        text = str(item.get("text", "")).strip()
        timestamps = item.get("timestamps") or item.get("timestamp")
        chars = _chars_from_timestamps(text, timestamps)
        start = chars[0].start if chars else 0.0
        end = chars[-1].end if chars else max(start + 0.001, start)
        segments.append(
            TranscriptSegment(
                id=f"seg_{item_index:04d}",
                start=start,
                end=end,
                text=text,
                chars=chars,
            )
        )

    if not segments:
        raise ASRError("FunASR returned no recognizable transcript segments.")

    return Transcript(source=str(source) if source else None, language="zh", segments=segments)


def _segments_from_sentence_info(
    sentence_info: list[object],
    offset: int = 0,
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for index, raw in enumerate(sentence_info, start=1):
        if not isinstance(raw, dict):
            continue

        text = str(raw.get("text", "")).strip()
        start = _timestamp_to_seconds(raw.get("start", 0))
        end = _timestamp_to_seconds(raw.get("end", start))
        if end <= start:
            end = start + 0.001

        speaker = raw.get("spk") if "spk" in raw else raw.get("speaker")
        segments.append(
            TranscriptSegment(
                id=f"seg_{offset + index:04d}",
                start=start,
                end=end,
                text=text,
                speaker=str(speaker) if speaker is not None else None,
            )
        )
    return segments


def _chars_from_timestamps(text: str, timestamps: object) -> list[TranscriptChar]:
    if not isinstance(timestamps, list):
        return []

    chars = [char for char in text if not char.isspace()]
    pairs = [item for item in timestamps if _is_timestamp_pair(item)]
    if not chars or not pairs:
        return []

    result = []
    for char, pair in zip(chars, pairs, strict=False):
        start = _timestamp_to_seconds(pair[0])
        end = _timestamp_to_seconds(pair[1])
        if end <= start:
            continue
        result.append(TranscriptChar(text=char, start=start, end=end))
    return result


def _is_timestamp_pair(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def _timestamp_to_seconds(value: object) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number / 1000 if abs(number) > 100 else number


def _ensure_ffmpeg_on_path() -> None:
    bin_dir = ffmpeg_bin_dir()
    if bin_dir is None:
        return

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    bin_dir_text = str(bin_dir)
    if bin_dir_text not in path_parts:
        os.environ["PATH"] = os.pathsep.join([bin_dir_text, *path_parts])
