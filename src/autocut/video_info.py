from typing import Any

from pydantic import BaseModel


class VideoInfo(BaseModel):
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None

    @classmethod
    def from_ffprobe(cls, payload: dict[str, Any]) -> "VideoInfo":
        streams = payload.get("streams", [])
        video_stream = next((item for item in streams if item.get("codec_type") == "video"), {})
        audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), {})
        fmt = payload.get("format", {})

        return cls(
            duration=_to_float(fmt.get("duration")),
            width=video_stream.get("width"),
            height=video_stream.get("height"),
            fps=_parse_fps(video_stream.get("r_frame_rate")),
            video_codec=video_stream.get("codec_name"),
            audio_codec=audio_stream.get("codec_name"),
        )


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_fps(value: str | None) -> float | None:
    if not value:
        return None
    if "/" not in value:
        return _to_float(value)

    numerator, denominator = value.split("/", 1)
    try:
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    except ValueError:
        return None

