import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class TranscriptChar(BaseModel):
    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> "TranscriptChar":
        if self.end <= self.start:
            raise ValueError(f"Character {self.text!r} end must be greater than start.")
        return self


class TranscriptSegment(BaseModel):
    id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str
    chars: list[TranscriptChar] = Field(default_factory=list)
    speaker: str | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> "TranscriptSegment":
        if self.end <= self.start:
            raise ValueError(f"Segment {self.id} end must be greater than start.")
        return self


class Transcript(BaseModel):
    source: str | None = None
    language: str = "zh"
    segments: list[TranscriptSegment]

    @field_validator("segments")
    @classmethod
    def validate_segments(cls, value: list[TranscriptSegment]) -> list[TranscriptSegment]:
        if not value:
            raise ValueError("Transcript must contain at least one segment.")
        return value

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments)


def load_transcript(path: Path) -> Transcript:
    if not path.exists():
        raise ValueError(f"Transcript file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Transcript.model_validate(data)


def save_transcript(transcript: Transcript, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")


def transcript_to_srt(transcript: Transcript) -> str:
    blocks = []
    for index, segment in enumerate(transcript.segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}",
                    segment.text,
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def write_srt(transcript: Transcript, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript_to_srt(transcript), encoding="utf-8")


def format_srt_time(seconds: float) -> str:
    milliseconds_total = round(seconds * 1000)
    hours, remainder = divmod(milliseconds_total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
