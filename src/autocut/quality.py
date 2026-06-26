import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from autocut.semantic import SentenceIndex


class TranscriptQualityReport(BaseModel):
    sentence_count: int = Field(ge=0)
    total_duration: float = Field(ge=0)
    max_sentence_duration: float = Field(ge=0)
    long_sentence_count: int = Field(ge=0)
    noise_token_ratio: float = Field(ge=0, le=1)
    chinese_char_ratio: float = Field(ge=0, le=1)
    repeated_char_run_count: int = Field(ge=0)
    score: float = Field(ge=0, le=1)
    recommendation: str
    reasons: list[str]


def assess_sentence_index(sentence_index: SentenceIndex) -> TranscriptQualityReport:
    sentences = sentence_index.sentences
    text = "".join(sentence.text for sentence in sentences)
    durations = [max(0.0, sentence.end - sentence.start) for sentence in sentences]
    total_duration = max((sentence.end for sentence in sentences), default=0.0) - min(
        (sentence.start for sentence in sentences), default=0.0
    )
    max_sentence_duration = max(durations, default=0.0)
    long_sentence_count = sum(1 for duration in durations if duration > 30)
    noise_token_ratio = _noise_token_ratio(text)
    chinese_char_ratio = _chinese_char_ratio(text)
    repeated_char_run_count = len(re.findall(r"(.)\1{3,}", text))

    penalties = [
        min(0.35, long_sentence_count * 0.035),
        min(0.25, noise_token_ratio * 1.2),
        0.20 if chinese_char_ratio < 0.55 else 0.0,
        min(0.15, repeated_char_run_count * 0.01),
        0.15 if max_sentence_duration > 90 else 0.0,
    ]
    score = round(max(0.0, 1.0 - sum(penalties)), 3)
    reasons = _build_reasons(
        max_sentence_duration=max_sentence_duration,
        long_sentence_count=long_sentence_count,
        noise_token_ratio=noise_token_ratio,
        chinese_char_ratio=chinese_char_ratio,
        repeated_char_run_count=repeated_char_run_count,
    )
    recommendation = (
        "prefer_subtitle_or_ocr"
        if score < 0.7 or long_sentence_count > 0 or max_sentence_duration > 60
        else "asr_usable"
    )

    return TranscriptQualityReport(
        sentence_count=len(sentences),
        total_duration=round(total_duration, 3),
        max_sentence_duration=round(max_sentence_duration, 3),
        long_sentence_count=long_sentence_count,
        noise_token_ratio=round(noise_token_ratio, 3),
        chinese_char_ratio=round(chinese_char_ratio, 3),
        repeated_char_run_count=repeated_char_run_count,
        score=score,
        recommendation=recommendation,
        reasons=reasons,
    )


def save_quality_report(report: TranscriptQualityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def load_quality_report(path: Path) -> TranscriptQualityReport:
    return TranscriptQualityReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _noise_token_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    noise_chars = sum(1 for char in chars if char in {"嗯", "啊", "唉", "呃", "额", "哦", "哈"})
    return noise_chars / len(chars)


def _chinese_char_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace() and not _is_punctuation(char)]
    if not chars:
        return 0.0
    chinese_chars = sum(1 for char in chars if "\u4e00" <= char <= "\u9fff")
    return chinese_chars / len(chars)


def _is_punctuation(char: str) -> bool:
    return bool(re.match(r"""[。！？!?；;、：:，,.…“”"'（）()《》]""", char))


def _build_reasons(
    *,
    max_sentence_duration: float,
    long_sentence_count: int,
    noise_token_ratio: float,
    chinese_char_ratio: float,
    repeated_char_run_count: int,
) -> list[str]:
    reasons = []
    if max_sentence_duration > 60:
        reasons.append("sentence_timestamps_have_large_spans")
    if long_sentence_count > 0:
        reasons.append("many_sentences_are_longer_than_30_seconds")
    if noise_token_ratio > 0.08:
        reasons.append("noise_tokens_are_frequent")
    if chinese_char_ratio < 0.55:
        reasons.append("recognized_text_has_low_chinese_character_ratio")
    if repeated_char_run_count > 0:
        reasons.append("recognized_text_contains_repeated_character_runs")
    return reasons
