import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from autocut.timeline import Timeline, TimelineSegment
from autocut.transcript import Transcript, TranscriptChar, TranscriptSegment, format_srt_time


EditDecision = Literal["keep", "delete"]


class ReservedFeatures(BaseModel):
    audio: dict | None = None
    visual: dict | None = None
    speaker: dict | None = None
    scene: dict | None = None


class SentenceItem(BaseModel):
    id: str
    source_segment_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str
    speaker: str | None = None
    pause_after: float = Field(default=0, ge=0)
    features: ReservedFeatures = Field(default_factory=ReservedFeatures)

    @model_validator(mode="after")
    def validate_time_range(self) -> "SentenceItem":
        if self.end <= self.start:
            raise ValueError(f"Sentence {self.id} end must be greater than start.")
        return self


class SentenceIndex(BaseModel):
    source: str | None = None
    sentences: list[SentenceItem]

    @field_validator("sentences")
    @classmethod
    def validate_sentences(cls, value: list[SentenceItem]) -> list[SentenceItem]:
        if not value:
            raise ValueError("Sentence index must contain at least one sentence.")
        return value


class TopicBlock(BaseModel):
    id: str
    start_sentence_id: str
    end_sentence_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str
    sentence_count: int = Field(gt=0)
    speaker: str | None = None
    features: ReservedFeatures = Field(default_factory=ReservedFeatures)

    @model_validator(mode="after")
    def validate_time_range(self) -> "TopicBlock":
        if self.end <= self.start:
            raise ValueError(f"Block {self.id} end must be greater than start.")
        return self


class TopicBlocks(BaseModel):
    source: str | None = None
    blocks: list[TopicBlock]

    @field_validator("blocks")
    @classmethod
    def validate_blocks(cls, value: list[TopicBlock]) -> list[TopicBlock]:
        if not value:
            raise ValueError("Topic blocks must contain at least one block.")
        return value


class ChapterPlan(BaseModel):
    title: str
    start_block_id: str
    end_block_id: str
    summary: str = ""


class BlockEditDecision(BaseModel):
    block_id: str
    decision: EditDecision
    reason: str = ""
    title: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)


class EditPlan(BaseModel):
    chapters: list[ChapterPlan] = Field(default_factory=list)
    edit_decisions: list[BlockEditDecision]

    @field_validator("edit_decisions")
    @classmethod
    def validate_decisions(cls, value: list[BlockEditDecision]) -> list[BlockEditDecision]:
        if not value:
            raise ValueError("Edit plan must contain at least one edit decision.")
        return value


def build_sentence_index(transcript: Transcript) -> SentenceIndex:
    sentences: list[SentenceItem] = []
    for segment in transcript.segments:
        sentence_parts = _split_segment_into_sentences(segment)
        for part in sentence_parts:
            sentences.append(
                SentenceItem(
                    id=f"S{len(sentences) + 1:04d}",
                    source_segment_id=segment.id,
                    start=part.start,
                    end=part.end,
                    text=part.text,
                    speaker=segment.speaker,
                )
            )

    for index, sentence in enumerate(sentences[:-1]):
        next_sentence = sentences[index + 1]
        sentence.pause_after = max(0, round(next_sentence.start - sentence.end, 3))

    return SentenceIndex(source=transcript.source, sentences=sentences)


class _SentencePart(BaseModel):
    start: float
    end: float
    text: str


def _split_segment_into_sentences(segment: TranscriptSegment) -> list[_SentencePart]:
    text = _normalize_text(segment.text)
    if not text:
        return []

    text_parts = _split_text_by_punctuation(text)
    if not text_parts:
        return [_SentencePart(start=segment.start, end=segment.end, text=text)]

    if not segment.chars:
        return _split_without_char_timestamps(segment, text_parts)

    return _split_with_char_timestamps(segment, text_parts)


def _split_text_by_punctuation(text: str) -> list[str]:
    parts = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", text)
    return [part for part in (_normalize_text(item) for item in parts) if part]


def _split_without_char_timestamps(
    segment: TranscriptSegment,
    text_parts: list[str],
) -> list[_SentencePart]:
    if len(text_parts) == 1:
        return [_SentencePart(start=segment.start, end=segment.end, text=text_parts[0])]

    duration = segment.end - segment.start
    total_chars = sum(len(part) for part in text_parts)
    cursor = segment.start
    result = []
    for index, part in enumerate(text_parts):
        if index == len(text_parts) - 1:
            end = segment.end
        else:
            end = cursor + duration * (len(part) / total_chars)
        result.append(_SentencePart(start=cursor, end=end, text=part))
        cursor = end
    return result


def _split_with_char_timestamps(
    segment: TranscriptSegment,
    text_parts: list[str],
) -> list[_SentencePart]:
    chars = _non_space_chars(segment.chars)
    cursor = 0
    result = []
    for part in text_parts:
        char_slice, cursor = _take_timed_chars(chars, cursor, part)
        if not char_slice:
            continue
        result.append(
            _SentencePart(
                start=char_slice[0].start,
                end=char_slice[-1].end,
                text=part,
            )
        )

    if result:
        return result
    return [_SentencePart(start=segment.start, end=segment.end, text=_normalize_text(segment.text))]


def _non_space_chars(chars: list[TranscriptChar]) -> list[TranscriptChar]:
    return [char for char in chars if char.text and not char.text.isspace()]


def _take_timed_chars(
    timed_chars: list[TranscriptChar],
    cursor: int,
    text_part: str,
) -> tuple[list[TranscriptChar], int]:
    char_slice: list[TranscriptChar] = []
    for text_char in text_part:
        if not text_char or text_char.isspace():
            continue
        if cursor >= len(timed_chars):
            break
        timed_char = timed_chars[cursor]
        if _chars_align(text_char, timed_char.text):
            char_slice.append(timed_char)
            cursor += 1
            continue
        if _is_punctuation(text_char):
            continue
        char_slice.append(timed_char)
        cursor += 1
    return char_slice, cursor


def _chars_align(text_char: str, timed_char: str) -> bool:
    return text_char == timed_char or (
        _is_punctuation(text_char) and _is_punctuation(timed_char)
    )


def _is_punctuation(char: str) -> bool:
    return bool(re.match(r"""[。！？!?；;、：:，,.…“”"'（）()《》]""", char))


def build_topic_blocks(
    sentence_index: SentenceIndex,
    min_duration: float = 20,
    max_duration: float = 60,
    split_pause: float = 1.2,
) -> TopicBlocks:
    blocks: list[TopicBlock] = []
    current: list[SentenceItem] = []

    for sentence in sentence_index.sentences:
        if current and _should_flush_block(current, sentence, min_duration, max_duration, split_pause):
            blocks.append(_make_block(current, len(blocks) + 1))
            current = []
        current.append(sentence)

    if current:
        blocks.append(_make_block(current, len(blocks) + 1))

    return TopicBlocks(source=sentence_index.source, blocks=blocks)


def compact_blocks_to_markdown(topic_blocks: TopicBlocks) -> str:
    lines = [
        "# Transcript Blocks For Editing",
        "",
        "Use block IDs when returning edit decisions. Do not invent timestamps.",
        "",
        "Expected output shape:",
        "",
        "```json",
        '{ "edit_decisions": [{ "block_id": "B0001", "decision": "keep", "reason": "" }] }',
        "```",
        "",
    ]

    for block in topic_blocks.blocks:
        start = format_srt_time(block.start).replace(",", ".")[:-1]
        end = format_srt_time(block.end).replace(",", ".")[:-1]
        speaker = f" | spk={block.speaker}" if block.speaker else ""
        lines.extend(
            [
                f"[{block.id} | {start}-{end} | {block.sentence_count} sentences{speaker}]",
                block.text,
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def edit_plan_to_timeline(edit_plan: EditPlan, topic_blocks: TopicBlocks) -> Timeline:
    block_by_id = {block.id: block for block in topic_blocks.blocks}
    segments: list[TimelineSegment] = []

    for decision in edit_plan.edit_decisions:
        block = block_by_id.get(decision.block_id)
        if block is None:
            raise ValueError(f"Edit plan references unknown block id: {decision.block_id}")

        segments.append(
            TimelineSegment(
                id=block.id,
                start=block.start,
                end=block.end,
                decision=decision.decision,
                text=decision.title or block.text,
                reason=decision.reason,
                confidence=decision.confidence,
            )
        )

    return Timeline(segments=segments)


def load_sentence_index(path: Path) -> SentenceIndex:
    return SentenceIndex.model_validate(_load_json(path, "Sentence index"))


def save_sentence_index(sentence_index: SentenceIndex, path: Path) -> None:
    _save_json(sentence_index, path)


def load_topic_blocks(path: Path) -> TopicBlocks:
    return TopicBlocks.model_validate(_load_json(path, "Topic blocks"))


def save_topic_blocks(topic_blocks: TopicBlocks, path: Path) -> None:
    _save_json(topic_blocks, path)


def load_edit_plan(path: Path) -> EditPlan:
    return EditPlan.model_validate(_load_json(path, "Edit plan"))


def save_compact_markdown(topic_blocks: TopicBlocks, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compact_blocks_to_markdown(topic_blocks), encoding="utf-8")


def sentence_index_to_srt(sentence_index: SentenceIndex) -> str:
    blocks = []
    for index, sentence in enumerate(sentence_index.sentences, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_time(sentence.start)} --> {format_srt_time(sentence.end)}",
                    sentence.text,
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def write_sentence_index_srt(sentence_index: SentenceIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sentence_index_to_srt(sentence_index), encoding="utf-8")


def _make_block(sentences: list[SentenceItem], index: int) -> TopicBlock:
    first = sentences[0]
    last = sentences[-1]
    speakers = {sentence.speaker for sentence in sentences if sentence.speaker}
    speaker = next(iter(speakers)) if len(speakers) == 1 else None
    return TopicBlock(
        id=f"B{index:04d}",
        start_sentence_id=first.id,
        end_sentence_id=last.id,
        start=first.start,
        end=last.end,
        text="".join(sentence.text for sentence in sentences),
        sentence_count=len(sentences),
        speaker=speaker,
    )


def _should_flush_block(
    current: list[SentenceItem],
    next_sentence: SentenceItem,
    min_duration: float,
    max_duration: float,
    split_pause: float,
) -> bool:
    first = current[0]
    last = current[-1]
    duration_with_next = next_sentence.end - first.start
    current_duration = last.end - first.start

    if current_duration >= min_duration and last.pause_after >= split_pause:
        return True
    if duration_with_next > max_duration:
        return True
    if current_duration >= min_duration and _looks_like_new_section(next_sentence.text):
        return True
    return False


def _looks_like_new_section(text: str) -> bool:
    return bool(
        re.match(
            r"^(首先|第一|第二|第三|第四|第五|接下来|然后|最后|总结|下面|我们来看|另一个)",
            text,
        )
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise ValueError(f"{label} file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _save_json(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
