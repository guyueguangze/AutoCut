from autocut.timeline import Timeline
from autocut.transcript import Transcript, TranscriptSegment


def transcript_to_preview_transcript(
    transcript: Transcript,
    timeline: Timeline,
    *,
    source: str | None = None,
) -> Transcript:
    output_segments: list[TranscriptSegment] = []
    output_cursor = 0.0

    for keep_segment in timeline.keep_segments():
        keep_duration = keep_segment.end - keep_segment.start
        for segment in transcript.segments:
            overlap_start = max(segment.start, keep_segment.start)
            overlap_end = min(segment.end, keep_segment.end)
            if overlap_end <= overlap_start:
                continue

            output_segments.append(
                TranscriptSegment(
                    id=f"preview_{len(output_segments) + 1:04d}",
                    start=round(output_cursor + (overlap_start - keep_segment.start), 3),
                    end=round(output_cursor + (overlap_end - keep_segment.start), 3),
                    text=segment.text,
                    speaker=segment.speaker,
                )
            )

        output_cursor += keep_duration

    if not output_segments:
        raise ValueError("Timeline has no subtitle segments to export.")

    return Transcript(
        source=source or transcript.source,
        language=transcript.language,
        segments=output_segments,
    )
