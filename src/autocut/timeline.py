import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Decision = Literal["keep", "delete"]


class TimelineSegment(BaseModel):
    id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    decision: Decision = "keep"
    text: str = ""
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_time_range(self) -> "TimelineSegment":
        if self.end <= self.start:
            raise ValueError(f"Segment {self.id} end must be greater than start.")
        return self


class Timeline(BaseModel):
    segments: list[TimelineSegment]

    @field_validator("segments")
    @classmethod
    def validate_segments(cls, value: list[TimelineSegment]) -> list[TimelineSegment]:
        if not value:
            raise ValueError("Timeline must contain at least one segment.")
        return value

    def keep_segments(self) -> list[TimelineSegment]:
        return sorted(
            [segment for segment in self.segments if segment.decision == "keep"],
            key=lambda segment: segment.start,
        )


def load_timeline(path: Path) -> Timeline:
    if not path.exists():
        raise ValueError(f"Timeline file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Timeline.model_validate(data)


def save_timeline(timeline: Timeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
