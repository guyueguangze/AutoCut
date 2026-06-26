from pathlib import Path

from autocut.semantic import EditPlan, TopicBlocks
from autocut.timeline import Timeline
from autocut.transcript import format_srt_time


def build_cut_report(
    *,
    job: str,
    topic_blocks: TopicBlocks,
    edit_plan: EditPlan,
    timeline: Timeline,
    render_output: Path | None = None,
) -> str:
    blocks_by_id = {block.id: block for block in topic_blocks.blocks}
    decided_ids = {decision.block_id for decision in edit_plan.edit_decisions}
    missing_decisions = [block.id for block in topic_blocks.blocks if block.id not in decided_ids]

    keep_segments = [segment for segment in timeline.segments if segment.decision == "keep"]
    delete_segments = [segment for segment in timeline.segments if segment.decision == "delete"]
    keep_duration = sum(segment.end - segment.start for segment in keep_segments)
    delete_duration = sum(segment.end - segment.start for segment in delete_segments)
    decided_duration = keep_duration + delete_duration
    source_span = _source_span(topic_blocks)
    compression = keep_duration / decided_duration if decided_duration > 0 else 0

    lines = [
        f"# Cut Report: {job}",
        "",
        "## Summary",
        "",
        f"- Source: {topic_blocks.source or 'unknown'}",
        f"- Topic blocks: {len(topic_blocks.blocks)}",
        f"- Decisions: {len(edit_plan.edit_decisions)}",
        f"- Keep segments: {len(keep_segments)}",
        f"- Delete segments: {len(delete_segments)}",
        f"- Source subtitle span: {_fmt_duration(source_span)}",
        f"- Decided duration: {_fmt_duration(decided_duration)}",
        f"- Estimated rough-cut duration: {_fmt_duration(keep_duration)}",
        f"- Removed duration: {_fmt_duration(delete_duration)}",
        f"- Keep ratio: {compression:.1%}",
    ]
    if render_output is not None:
        lines.append(f"- Render output: {render_output}")

    warnings = _warnings(keep_segments, missing_decisions)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    if edit_plan.chapters:
        lines.extend(["", "## Chapters", ""])
        for chapter in edit_plan.chapters:
            lines.append(
                f"- {chapter.title}: {chapter.start_block_id} -> {chapter.end_block_id}"
                + (f" | {chapter.summary}" if chapter.summary else "")
            )

    lines.extend(["", "## Keep Segments", ""])
    for segment in keep_segments:
        block = blocks_by_id.get(segment.id)
        text = _preview_text(block.text if block else segment.text)
        lines.append(
            f"- {segment.id} | {_fmt_time(segment.start)}-{_fmt_time(segment.end)} "
            f"| {_fmt_duration(segment.end - segment.start)} | {segment.text}"
        )
        if segment.reason:
            lines.append(f"  Reason: {segment.reason}")
        if text:
            lines.append(f"  Text: {text}")

    if delete_segments:
        lines.extend(["", "## Delete Segments", ""])
        for segment in delete_segments:
            block = blocks_by_id.get(segment.id)
            text = _preview_text(block.text if block else segment.text)
            lines.append(
                f"- {segment.id} | {_fmt_time(segment.start)}-{_fmt_time(segment.end)} "
                f"| {_fmt_duration(segment.end - segment.start)} | {segment.text}"
            )
            if segment.reason:
                lines.append(f"  Reason: {segment.reason}")
            if text:
                lines.append(f"  Text: {text}")

    if missing_decisions:
        lines.extend(["", "## Missing Decisions", ""])
        lines.extend(f"- {block_id}" for block_id in missing_decisions)

    return "\n".join(lines).rstrip() + "\n"


def save_cut_report(report: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _source_span(topic_blocks: TopicBlocks) -> float:
    if not topic_blocks.blocks:
        return 0.0
    return max(block.end for block in topic_blocks.blocks) - min(
        block.start for block in topic_blocks.blocks
    )


def _warnings(keep_segments, missing_decisions: list[str]) -> list[str]:
    warnings = []
    if not keep_segments:
        warnings.append("No keep segments; render output would be empty.")
    if missing_decisions:
        warnings.append(f"{len(missing_decisions)} blocks have no keep/delete decision.")
    long_segments = [
        segment.id for segment in keep_segments if segment.end - segment.start > 90
    ]
    if long_segments:
        warnings.append(f"Long keep segments over 90s: {', '.join(long_segments)}.")
    return warnings


def _fmt_time(seconds: float) -> str:
    return format_srt_time(seconds).replace(",", ".")[:-4]


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _preview_text(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."
