import re
from dataclasses import dataclass
from pathlib import Path

import typer

from autocut.ffmpeg import (
    FFmpegError,
    extract_audio,
    extract_frame,
    extract_frames,
    iter_frames,
    probe_subtitle_streams,
    probe_video,
    render_timeline,
)
from autocut.timeline import load_timeline, save_timeline
from autocut.transcript import load_transcript, save_transcript, write_srt

app = typer.Typer(help="Chinese speech-first automatic video rough-cut MVP.")


@dataclass(frozen=True)
class JobPaths:
    job: str
    run_dir: Path
    audio: Path
    transcript: Path
    subtitle_srt: Path
    sentence_index: Path
    topic_blocks: Path
    llm_input: Path
    edit_plan: Path
    timeline: Path
    cut_report: Path
    preview_srt: Path
    render_output: Path
    ocr_work_dir: Path


def _safe_job_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\s]+', "_", value.strip())
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "default"


def _job_paths(job: str) -> JobPaths:
    safe_job = _safe_job_name(job)
    run_dir = Path("runs") / safe_job
    return JobPaths(
        job=safe_job,
        run_dir=run_dir,
        audio=run_dir / "audio.wav",
        transcript=run_dir / "transcript.json",
        subtitle_srt=Path("outputs") / f"{safe_job}.subtitles.srt",
        sentence_index=run_dir / "sentence_index.json",
        topic_blocks=run_dir / "topic_blocks.json",
        llm_input=run_dir / "llm_input.md",
        edit_plan=run_dir / "edit_plan.json",
        timeline=run_dir / "timeline.json",
        cut_report=run_dir / "cut_report.md",
        preview_srt=Path("outputs") / f"{safe_job}.rough_cut.srt",
        render_output=Path("outputs") / f"{safe_job}.rough_cut.mp4",
        ocr_work_dir=run_dir / "ocr_work",
    )


def _default_job_name(video: Path) -> str:
    return _safe_job_name(video.stem)


@app.command()
def doctor() -> None:
    """Check required local tools."""
    from autocut.ffmpeg import check_ffmpeg

    status = check_ffmpeg()
    if status.ok:
        typer.echo("OK: ffmpeg and ffprobe are available.")
        return

    typer.echo("Missing required tools:")
    for item in status.missing:
        typer.echo(f"- {item}")
    raise typer.Exit(1)


@app.command()
def probe(video: Path) -> None:
    """Print basic video metadata."""
    try:
        info = probe_video(video)
    except FFmpegError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(info.model_dump_json(indent=2))


@app.command("probe-subtitles")
def probe_subtitles(video: Path) -> None:
    """Print embedded subtitle streams, if any."""
    try:
        streams = probe_subtitle_streams(video)
    except FFmpegError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    if not streams:
        typer.echo("No embedded subtitle streams found.")
        return

    for stream in streams:
        language = stream.language or "unknown"
        title = f" title={stream.title}" if stream.title else ""
        typer.echo(
            f"stream={stream.index} codec={stream.codec_name or 'unknown'} lang={language}{title}"
        )


@app.command("run")
def run_pipeline(
    video: Path,
    job: str | None = typer.Option(None, "--job", "-j"),
    mode: str = typer.Option("ocr", "--mode", help="Recognition mode: ocr or asr."),
    render_output: bool = typer.Option(False, "--render/--no-render"),
    crop: str = typer.Option("768:90:0:486", "--crop"),
    detect_interval: float = typer.Option(1.0, "--detect-interval"),
    refine_interval: float = typer.Option(1.0, "--refine-interval"),
    expand: float = typer.Option(1.0, "--expand"),
    workers: int = typer.Option(4, "--workers"),
    ocr_threads: int | None = typer.Option(1, "--ocr-threads"),
    ocr_engine: str = typer.Option("rapidocr", "--ocr-engine"),
    ppocrv6_model_size: str = typer.Option("tiny", "--ppocrv6-model-size"),
    ocr_device: str | None = typer.Option(None, "--ocr-device"),
    ocr_inference_engine: str | None = typer.Option(None, "--ocr-inference-engine"),
    signature_threshold: int = typer.Option(128, "--signature-threshold"),
    min_score: float = typer.Option(0.7, "--min-score"),
    sample_rate: int = typer.Option(16000, "--sample-rate"),
    asr_device: str = typer.Option("cpu", "--asr-device"),
) -> None:
    """Run the MVP pipeline for one video into runs/<job>/."""
    from autocut.asr import ASRError, FunASRAdapter
    from autocut.semantic import (
        build_sentence_index,
        build_topic_blocks,
        edit_plan_to_timeline,
        load_edit_plan,
        load_topic_blocks,
        save_compact_markdown,
        save_sentence_index,
        save_topic_blocks,
    )

    if not video.exists():
        typer.echo(f"Video file does not exist: {video}")
        raise typer.Exit(1)

    mode = mode.lower().strip()
    if mode not in {"ocr", "asr"}:
        typer.echo("Mode must be either 'ocr' or 'asr'.")
        raise typer.Exit(1)

    paths = _job_paths(job or _default_job_name(video))
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Job: {paths.job}")
    typer.echo(f"Run dir: {paths.run_dir}")

    try:
        if mode == "ocr":
            ocr_subtitles_adaptive(
                video,
                output=paths.subtitle_srt,
                transcript_output=paths.transcript,
                work_dir=paths.ocr_work_dir,
                detect_interval=detect_interval,
                refine_interval=refine_interval,
                crop=crop,
                start=None,
                duration=None,
                expand=expand,
                min_score=min_score,
                signature_threshold=signature_threshold,
                min_white_pixels=120,
                workers=workers,
                ocr_threads=ocr_threads,
                ocr_engine=ocr_engine,
                ppocrv6_model_size=ppocrv6_model_size,
                ocr_device=ocr_device,
                ocr_inference_engine=ocr_inference_engine,
                stream=True,
                rec_only=True,
                cache=False,
                single_refine_stream=True,
            )
        else:
            extract_audio(video, paths.audio, sample_rate=sample_rate)
            adapter = FunASRAdapter(device=asr_device)
            transcript_model = adapter.transcribe(paths.audio, source=video)
            save_transcript(transcript_model, paths.transcript)
            write_srt(transcript_model, paths.subtitle_srt)
            typer.echo(f"Transcript written to {paths.transcript}")
            typer.echo(f"SRT written to {paths.subtitle_srt}")

        transcript_model = load_transcript(paths.transcript)
        sentence_index = build_sentence_index(transcript_model)
        save_sentence_index(sentence_index, paths.sentence_index)

        topic_blocks = build_topic_blocks(sentence_index)
        save_topic_blocks(topic_blocks, paths.topic_blocks)
        save_compact_markdown(topic_blocks, paths.llm_input)

        typer.echo(f"Sentence index written to {paths.sentence_index}")
        typer.echo(f"Topic blocks written to {paths.topic_blocks}")
        typer.echo(f"LLM input written to {paths.llm_input}")

        if paths.edit_plan.exists():
            from autocut.report import build_cut_report, save_cut_report

            edit_plan = load_edit_plan(paths.edit_plan)
            topic_blocks_model = load_topic_blocks(paths.topic_blocks)
            timeline = edit_plan_to_timeline(edit_plan, topic_blocks_model)
            save_timeline(timeline, paths.timeline)
            report = build_cut_report(
                job=paths.job,
                topic_blocks=topic_blocks_model,
                edit_plan=edit_plan,
                timeline=timeline,
                render_output=paths.render_output if render_output else None,
            )
            save_cut_report(report, paths.cut_report)
            typer.echo(f"Timeline written to {paths.timeline}")
            typer.echo(f"Cut report written to {paths.cut_report}")
            if paths.transcript.exists():
                _write_preview_srt(paths, timeline)
                typer.echo(f"Preview SRT written to {paths.preview_srt}")
            if render_output:
                render_timeline(
                    video,
                    timeline.keep_segments(),
                    paths.render_output,
                    reencode=True,
                )
                typer.echo(f"Rough cut written to {paths.render_output}")
        else:
            typer.echo(f"Next: send {paths.llm_input} to an LLM, then save {paths.edit_plan}.")
    except (FFmpegError, ASRError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc


@app.command("apply-plan")
def apply_plan_cmd(
    job: str = typer.Option(..., "--job", "-j"),
    edit_plan: Path | None = typer.Option(None, "--edit-plan"),
    render_output: bool = typer.Option(False, "--render/--no-render"),
    video: Path | None = typer.Option(None, "--video"),
    reencode: bool = typer.Option(True, "--reencode/--stream-copy"),
) -> None:
    """Apply runs/<job>/edit_plan.json and write timeline.json."""
    from autocut.report import build_cut_report, save_cut_report
    from autocut.semantic import edit_plan_to_timeline, load_edit_plan, load_topic_blocks

    paths = _job_paths(job)
    plan_path = edit_plan or paths.edit_plan

    if not paths.topic_blocks.exists():
        typer.echo(f"Topic blocks file does not exist: {paths.topic_blocks}")
        raise typer.Exit(1)
    if not plan_path.exists():
        typer.echo(f"Edit plan file does not exist: {plan_path}")
        raise typer.Exit(1)
    if render_output and video is None:
        typer.echo("--video is required when using --render.")
        raise typer.Exit(1)

    try:
        edit_plan_model = load_edit_plan(plan_path)
        topic_blocks_model = load_topic_blocks(paths.topic_blocks)
        timeline = edit_plan_to_timeline(edit_plan_model, topic_blocks_model)
        save_timeline(timeline, paths.timeline)
        report = build_cut_report(
            job=paths.job,
            topic_blocks=topic_blocks_model,
            edit_plan=edit_plan_model,
            timeline=timeline,
            render_output=paths.render_output if render_output else None,
        )
        save_cut_report(report, paths.cut_report)

        typer.echo(f"Timeline written to {paths.timeline}")
        typer.echo(f"Cut report written to {paths.cut_report}")
        if paths.transcript.exists():
            _write_preview_srt(paths, timeline)
            typer.echo(f"Preview SRT written to {paths.preview_srt}")

        if render_output:
            assert video is not None
            render_timeline(video, timeline.keep_segments(), paths.render_output, reencode=reencode)
            typer.echo(f"Rough cut written to {paths.render_output}")
    except (FFmpegError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc


@app.command("report")
def report_cmd(
    job: str = typer.Option(..., "--job", "-j"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    render_output: bool = typer.Option(False, "--render-output/--no-render-output"),
) -> None:
    """Build a pre-render cut report from edit_plan.json and timeline.json."""
    from autocut.report import build_cut_report, save_cut_report
    from autocut.semantic import load_edit_plan, load_topic_blocks

    paths = _job_paths(job)
    report_output = output or paths.cut_report
    if not paths.topic_blocks.exists():
        typer.echo(f"Topic blocks file does not exist: {paths.topic_blocks}")
        raise typer.Exit(1)
    if not paths.edit_plan.exists():
        typer.echo(f"Edit plan file does not exist: {paths.edit_plan}")
        raise typer.Exit(1)
    if not paths.timeline.exists():
        typer.echo(f"Timeline file does not exist: {paths.timeline}")
        raise typer.Exit(1)

    try:
        topic_blocks = load_topic_blocks(paths.topic_blocks)
        edit_plan = load_edit_plan(paths.edit_plan)
        timeline = load_timeline(paths.timeline)
        report = build_cut_report(
            job=paths.job,
            topic_blocks=topic_blocks,
            edit_plan=edit_plan,
            timeline=timeline,
            render_output=paths.render_output if render_output else None,
        )
        save_cut_report(report, report_output)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Cut report written to {report_output}")


@app.command("preview-srt")
def preview_srt_cmd(
    job: str = typer.Option(..., "--job", "-j"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Export subtitles remapped to the rough-cut timeline."""
    paths = _job_paths(job)
    output_path = output or paths.preview_srt

    if not paths.transcript.exists():
        typer.echo(f"Transcript file does not exist: {paths.transcript}")
        raise typer.Exit(1)
    if not paths.timeline.exists():
        typer.echo(f"Timeline file does not exist: {paths.timeline}")
        raise typer.Exit(1)

    try:
        timeline = load_timeline(paths.timeline)
        _write_preview_srt(paths, timeline, output=output_path)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Preview SRT written to {output_path}")


def _write_preview_srt(
    paths: JobPaths,
    timeline,
    *,
    output: Path | None = None,
) -> None:
    from autocut.preview import transcript_to_preview_transcript

    transcript = load_transcript(paths.transcript)
    preview_transcript = transcript_to_preview_transcript(
        transcript,
        timeline,
        source=str(paths.render_output),
    )
    write_srt(preview_transcript, output or paths.preview_srt)


@app.command("extract-audio")
def extract_audio_cmd(
    video: Path,
    output: Path = typer.Option(Path("runs/default/audio.wav"), "--output", "-o"),
    sample_rate: int = typer.Option(16000, "--sample-rate"),
) -> None:
    """Extract mono WAV audio for ASR."""
    try:
        extract_audio(video, output, sample_rate=sample_rate)
    except FFmpegError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Audio written to {output}")


@app.command("ocr-frame")
def ocr_frame(
    video: Path,
    timestamp: float = typer.Option(..., "--timestamp", "-t"),
    output: Path = typer.Option(Path("runs/default/ocr/frame.json"), "--output", "-o"),
    image_output: Path = typer.Option(Path("runs/default/ocr/frame.png"), "--image-output"),
    crop: str | None = typer.Option(
        None,
        "--crop",
        help="FFmpeg crop expression, for example 768:180:0:396.",
    ),
    ocr_engine: str = typer.Option("rapidocr", "--ocr-engine"),
    ppocrv6_model_size: str = typer.Option("tiny", "--ppocrv6-model-size"),
    ocr_device: str | None = typer.Option(None, "--ocr-device"),
    ocr_inference_engine: str | None = typer.Option(None, "--ocr-inference-engine"),
) -> None:
    """Extract one frame and run OCR on it."""
    from autocut.ocr import ocr_image

    try:
        extract_frame(video, image_output, timestamp=timestamp, crop=crop)
        result = ocr_image(
            image_output,
            timestamp=timestamp,
            engine_name=ocr_engine,
            model_size=ppocrv6_model_size,
            device=ocr_device,
            inference_engine=ocr_inference_engine,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except (FFmpegError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(result.model_dump_json(indent=2))
    typer.echo(f"OCR result written to {output}")


@app.command("ocr-subtitles")
def ocr_subtitles(
    video: Path,
    output: Path = typer.Option(Path("outputs/ocr_subtitles.srt"), "--output", "-o"),
    transcript_output: Path = typer.Option(
        Path("runs/default/ocr_subtitles.transcript.json"),
        "--transcript-output",
    ),
    work_dir: Path = typer.Option(Path("runs/default/ocr_frames"), "--work-dir"),
    interval: float = typer.Option(1.0, "--interval"),
    crop: str = typer.Option("768:180:0:396", "--crop"),
    start: float | None = typer.Option(None, "--start"),
    duration: float | None = typer.Option(None, "--duration"),
    min_score: float = typer.Option(0.7, "--min-score"),
    fast: bool = typer.Option(True, "--fast/--no-fast"),
    signature_threshold: int = typer.Option(16, "--signature-threshold"),
    min_white_pixels: int = typer.Option(120, "--min-white-pixels"),
    workers: int = typer.Option(1, "--workers"),
    ocr_threads: int | None = typer.Option(None, "--ocr-threads"),
    ocr_engine: str = typer.Option("rapidocr", "--ocr-engine"),
    ppocrv6_model_size: str = typer.Option("tiny", "--ppocrv6-model-size"),
    ocr_device: str | None = typer.Option(None, "--ocr-device"),
    ocr_inference_engine: str | None = typer.Option(None, "--ocr-inference-engine"),
) -> None:
    """OCR hardcoded subtitles into SRT and transcript JSON."""
    from autocut.ocr import (
        filter_trailing_credit_segments,
        merge_ocr_samples,
        ocr_segments_to_transcript,
    )
    from autocut.transcript import save_transcript, write_srt

    if interval <= 0:
        typer.echo("Interval must be greater than 0.")
        raise typer.Exit(1)
    if workers <= 0:
        typer.echo("Workers must be greater than 0.")
        raise typer.Exit(1)
    if ocr_threads is not None and ocr_threads <= 0:
        typer.echo("OCR threads must be greater than 0.")
        raise typer.Exit(1)

    try:
        from autocut.ocr import ensure_ocr_engine_available

        ensure_ocr_engine_available(ocr_engine)
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    try:
        frame_pattern = work_dir / "frame_%06d.png"
        work_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in work_dir.glob("frame_*.png"):
            old_frame.unlink()
        extract_frames(
            video,
            frame_pattern,
            fps=1 / interval,
            crop=crop,
            start=start,
            duration=duration,
        )
        frame_paths = sorted(work_dir.glob("frame_*.png"))
        base_time = start or 0.0
        complete_samples, stats = _ocr_frame_paths_to_samples(
            frame_paths,
            base_time=base_time,
            interval=interval,
            min_score=min_score,
            fast=fast,
            signature_threshold=signature_threshold,
            min_white_pixels=min_white_pixels,
            workers=workers,
            ocr_threads=ocr_threads,
            ocr_engine=ocr_engine,
            ppocrv6_model_size=ppocrv6_model_size,
            ocr_device=ocr_device,
            ocr_inference_engine=ocr_inference_engine,
        )
        segments = merge_ocr_samples(complete_samples, interval=interval)
        segments = filter_trailing_credit_segments(segments)
        transcript_model = ocr_segments_to_transcript(segments, source=str(video))
        save_transcript(transcript_model, transcript_output)
        write_srt(transcript_model, output)
    except (FFmpegError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"OCR subtitle segments: {len(segments)}")
    typer.echo(
        f"Frames: {len(frame_paths)}, OCR calls: {stats['ocr_calls']}, "
        f"skipped: {stats['skipped']}, reused: {stats['reused']}, "
        f"workers: {workers}, ocr_threads: {ocr_threads or 'auto'}, "
        f"ocr_engine: {ocr_engine}, ppocrv6_model_size: {ppocrv6_model_size}"
    )
    typer.echo(f"SRT written to {output}")
    typer.echo(f"Transcript written to {transcript_output}")


@app.command("ocr-subtitles-adaptive")
def ocr_subtitles_adaptive(
    video: Path,
    output: Path = typer.Option(Path("outputs/ocr_subtitles.adaptive.srt"), "--output", "-o"),
    transcript_output: Path = typer.Option(
        Path("runs/default/ocr_subtitles.adaptive.transcript.json"),
        "--transcript-output",
    ),
    work_dir: Path = typer.Option(Path("runs/default/ocr_adaptive"), "--work-dir"),
    detect_interval: float = typer.Option(1.0, "--detect-interval"),
    refine_interval: float = typer.Option(1.0, "--refine-interval"),
    crop: str = typer.Option("768:90:0:486", "--crop"),
    start: float | None = typer.Option(None, "--start"),
    duration: float | None = typer.Option(None, "--duration"),
    expand: float = typer.Option(1.0, "--expand"),
    min_score: float = typer.Option(0.7, "--min-score"),
    signature_threshold: int = typer.Option(128, "--signature-threshold"),
    min_white_pixels: int = typer.Option(120, "--min-white-pixels"),
    workers: int = typer.Option(4, "--workers"),
    ocr_threads: int | None = typer.Option(1, "--ocr-threads"),
    ocr_engine: str = typer.Option("rapidocr", "--ocr-engine"),
    ppocrv6_model_size: str = typer.Option("tiny", "--ppocrv6-model-size"),
    ocr_device: str | None = typer.Option(None, "--ocr-device"),
    ocr_inference_engine: str | None = typer.Option(None, "--ocr-inference-engine"),
    ocr_batch_size: int = typer.Option(1, "--ocr-batch-size"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    rec_only: bool = typer.Option(True, "--rec-only/--full-ocr"),
    cache: bool = typer.Option(False, "--cache/--no-cache"),
    single_refine_stream: bool = typer.Option(
        True,
        "--single-refine-stream/--range-refine-stream",
    ),
) -> None:
    """Detect subtitle ranges first, then OCR only those ranges."""
    from autocut.ocr import (
        detect_subtitle_ranges,
        filter_trailing_credit_segments,
        merge_ocr_samples,
        ocr_segments_to_transcript,
        subtitle_frame_signature,
        subtitle_frame_signature_from_array,
    )
    from autocut.transcript import save_transcript, write_srt

    if detect_interval <= 0 or refine_interval <= 0:
        typer.echo("Intervals must be greater than 0.")
        raise typer.Exit(1)
    if workers <= 0:
        typer.echo("Workers must be greater than 0.")
        raise typer.Exit(1)
    if ocr_threads is not None and ocr_threads <= 0:
        typer.echo("OCR threads must be greater than 0.")
        raise typer.Exit(1)
    if ocr_batch_size <= 0:
        typer.echo("OCR batch size must be greater than 0.")
        raise typer.Exit(1)

    try:
        from autocut.ocr import ensure_ocr_engine_available

        ensure_ocr_engine_available(ocr_engine)
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    try:
        base_time = start or 0.0
        end_bound = base_time + duration if duration is not None else probe_video(video).duration
        detections = []
        if stream:
            detect_frames_count = 0
            for index, frame in enumerate(
                iter_frames(
                    video,
                    fps=1 / detect_interval,
                    crop=crop,
                    start=start,
                    duration=duration,
                )
            ):
                timestamp = base_time + index * detect_interval
                signature = subtitle_frame_signature_from_array(
                    frame,
                    min_white_pixels=min_white_pixels,
                )
                detections.append((timestamp, signature is not None))
                detect_frames_count += 1
        else:
            detect_dir = work_dir / "detect"
            _clear_frame_dir(detect_dir)
            extract_frames(
                video,
                detect_dir / "frame_%06d.png",
                fps=1 / detect_interval,
                crop=crop,
                start=start,
                duration=duration,
            )
            detect_frames = sorted(detect_dir.glob("frame_*.png"))
            detect_frames_count = len(detect_frames)
            for index, frame_path in enumerate(detect_frames):
                timestamp = base_time + index * detect_interval
                signature = subtitle_frame_signature(
                    frame_path,
                    min_white_pixels=min_white_pixels,
                )
                detections.append((timestamp, signature is not None))

        ranges = detect_subtitle_ranges(
            detections,
            interval=detect_interval,
            expand=expand,
            start_bound=base_time,
            end_bound=end_bound,
        )

        all_samples = []
        all_sample_slots = []
        all_sample_timestamps = []
        all_array_tasks = []
        all_copy_sources = {}
        total_frames = 0
        total_stats = {"ocr_calls": 0, "skipped": 0, "reused": 0, "cached": 0}
        global_signature_sources: dict[bytes, int] | None = {} if cache else None
        if stream and single_refine_stream:
            frames = _select_frames_in_ranges(
                iter_frames(
                    video,
                    fps=1 / refine_interval,
                    crop=crop,
                    start=base_time,
                    duration=(end_bound - base_time) if end_bound is not None else duration,
                ),
                base_time=base_time,
                interval=refine_interval,
                ranges=ranges,
            )
            sample_slots, timestamps, tasks, copy_sources, stats = _plan_ocr_frame_arrays(
                frames,
                sample_offset=0,
                min_score=min_score,
                signature_threshold=signature_threshold,
                min_white_pixels=min_white_pixels,
                rec_only=rec_only,
                cache_sources=global_signature_sources,
            )
            all_sample_slots.extend(sample_slots)
            all_sample_timestamps.extend(timestamps)
            all_array_tasks.extend(tasks)
            all_copy_sources.update(copy_sources)
            total_frames += len(frames)
            for key in total_stats:
                total_stats[key] += stats[key]
        else:
            for range_index, time_range in enumerate(ranges, start=1):
                if stream:
                    frames = [
                        (time_range.start + index * refine_interval, frame)
                        for index, frame in enumerate(
                            iter_frames(
                                video,
                                fps=1 / refine_interval,
                                crop=crop,
                                start=time_range.start,
                                duration=time_range.end - time_range.start,
                            )
                        )
                    ]
                    sample_slots, timestamps, tasks, copy_sources, stats = _plan_ocr_frame_arrays(
                        frames,
                        sample_offset=len(all_sample_slots),
                        min_score=min_score,
                        signature_threshold=signature_threshold,
                        min_white_pixels=min_white_pixels,
                        rec_only=rec_only,
                        cache_sources=global_signature_sources,
                    )
                    all_sample_slots.extend(sample_slots)
                    all_sample_timestamps.extend(timestamps)
                    all_array_tasks.extend(tasks)
                    all_copy_sources.update(copy_sources)
                    frame_count = len(frames)
                else:
                    refine_dir = work_dir / f"refine_{range_index:04d}"
                    _clear_frame_dir(refine_dir)
                    extract_frames(
                        video,
                        refine_dir / "frame_%06d.png",
                        fps=1 / refine_interval,
                        crop=crop,
                        start=time_range.start,
                        duration=time_range.end - time_range.start,
                    )
                    refine_frames = sorted(refine_dir.glob("frame_*.png"))
                    samples, stats = _ocr_frame_paths_to_samples(
                        refine_frames,
                        base_time=time_range.start,
                        interval=refine_interval,
                        min_score=min_score,
                        fast=True,
                        signature_threshold=signature_threshold,
                        min_white_pixels=min_white_pixels,
                        workers=workers,
                        ocr_threads=ocr_threads,
                        ocr_engine=ocr_engine,
                        ppocrv6_model_size=ppocrv6_model_size,
                        ocr_device=ocr_device,
                        ocr_inference_engine=ocr_inference_engine,
                    )
                    frame_count = len(refine_frames)
                    all_samples.extend(samples)
                total_frames += frame_count
                for key in total_stats:
                    total_stats[key] += stats[key]

        if stream:
            from autocut.ocr import OCRSample, run_ocr_array_tasks_batched

            for index, sample in run_ocr_array_tasks_batched(
                all_array_tasks,
                workers=workers,
                ocr_threads=ocr_threads,
                engine_name=ocr_engine,
                model_size=ppocrv6_model_size,
                device=ocr_device,
                inference_engine=ocr_inference_engine,
                batch_size=ocr_batch_size,
            ):
                all_sample_slots[index] = sample

            for index, source_index in all_copy_sources.items():
                source = all_sample_slots[source_index] or OCRSample(
                    timestamp=all_sample_timestamps[source_index]
                )
                all_sample_slots[index] = OCRSample(
                    timestamp=all_sample_timestamps[index],
                    text=source.text,
                    score=source.score,
                )

            all_samples = [
                sample if sample is not None else OCRSample(timestamp=all_sample_timestamps[index])
                for index, sample in enumerate(all_sample_slots)
            ]

        all_samples.sort(key=lambda item: item.timestamp)
        segments = merge_ocr_samples(all_samples, interval=refine_interval)
        segments = filter_trailing_credit_segments(segments)
        transcript_model = ocr_segments_to_transcript(segments, source=str(video))
        save_transcript(transcript_model, transcript_output)
        write_srt(transcript_model, output)
    except (FFmpegError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Detected ranges: {len(ranges)}")
    typer.echo(f"OCR subtitle segments: {len(segments)}")
    typer.echo(
        f"Detect frames: {detect_frames_count}, refine frames: {total_frames}, "
        f"OCR calls: {total_stats['ocr_calls']}, skipped: {total_stats['skipped']}, "
        f"reused: {total_stats['reused']}, cached: {total_stats['cached']}, workers: {workers}, "
        f"ocr_threads: {ocr_threads or 'auto'}, stream: {stream}, "
        f"ocr_engine: {ocr_engine}, ppocrv6_model_size: {ppocrv6_model_size}, "
        f"ocr_batch_size: {ocr_batch_size}, rec_only: {rec_only}, cache: {cache}, "
        f"single_refine_stream: {single_refine_stream}"
    )
    typer.echo(f"SRT written to {output}")
    typer.echo(f"Transcript written to {transcript_output}")


def _clear_frame_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old_frame in directory.glob("frame_*.png"):
        old_frame.unlink()


def _select_frames_in_ranges(
    frames,
    *,
    base_time: float,
    interval: float,
    ranges,
) -> list[tuple[float, object]]:
    selected = []
    range_index = 0
    for frame_index, frame in enumerate(frames):
        timestamp = base_time + frame_index * interval
        while range_index < len(ranges) and timestamp >= ranges[range_index].end:
            range_index += 1
        if range_index >= len(ranges):
            break
        current_range = ranges[range_index]
        if current_range.start <= timestamp < current_range.end:
            selected.append((timestamp, frame))
    return selected


def _ocr_frame_paths_to_samples(
    frame_paths: list[Path],
    *,
    base_time: float,
    interval: float,
    min_score: float,
    fast: bool,
    signature_threshold: int,
    min_white_pixels: int,
    workers: int,
    ocr_threads: int | None,
    ocr_engine: str = "rapidocr",
    ppocrv6_model_size: str = "tiny",
    ocr_device: str | None = None,
    ocr_inference_engine: str | None = None,
) -> tuple[list["OCRSample"], dict[str, int]]:
    from autocut.ocr import OCRSample, run_ocr_tasks, signature_distance, subtitle_frame_signature

    samples: list[OCRSample | None] = [None] * len(frame_paths)
    ocr_tasks = []
    copy_sources: dict[int, int] = {}
    stats = {"ocr_calls": 0, "skipped": 0, "reused": 0, "cached": 0}
    previous_signature = None
    previous_ocr_source: int | None = None

    for index, frame_path in enumerate(frame_paths):
        timestamp = base_time + index * interval
        if fast:
            signature = subtitle_frame_signature(
                frame_path,
                min_white_pixels=min_white_pixels,
            )
            if signature is None:
                samples[index] = OCRSample(timestamp=timestamp)
                previous_signature = None
                previous_ocr_source = None
                stats["skipped"] += 1
                continue

            distance = signature_distance(previous_signature, signature)
            if (
                distance is not None
                and distance <= signature_threshold
                and previous_ocr_source is not None
            ):
                copy_sources[index] = previous_ocr_source
                previous_signature = signature
                stats["reused"] += 1
                continue
        else:
            signature = None

        ocr_tasks.append((index, str(frame_path), timestamp, min_score))
        previous_signature = signature
        previous_ocr_source = index
        stats["ocr_calls"] += 1

    for index, sample in run_ocr_tasks(
        ocr_tasks,
        workers=workers,
        ocr_threads=ocr_threads,
        engine_name=ocr_engine,
        model_size=ppocrv6_model_size,
        device=ocr_device,
        inference_engine=ocr_inference_engine,
    ):
        samples[index] = sample

    for index, source_index in copy_sources.items():
        source = samples[source_index] or OCRSample(timestamp=base_time + source_index * interval)
        samples[index] = OCRSample(
            timestamp=base_time + index * interval,
            text=source.text,
            score=source.score,
        )

    complete_samples = [
        sample if sample is not None else OCRSample(timestamp=base_time + index * interval)
        for index, sample in enumerate(samples)
    ]
    return complete_samples, stats


def _ocr_frame_arrays_to_samples(
    frames: list[tuple[float, object]],
    *,
    min_score: float,
    signature_threshold: int,
    min_white_pixels: int,
    workers: int,
    ocr_threads: int | None,
    ocr_engine: str,
    ppocrv6_model_size: str,
    ocr_device: str | None,
    ocr_inference_engine: str | None,
    ocr_batch_size: int = 16,
    rec_only: bool = True,
    cache_sources: dict[bytes, int] | None = None,
) -> tuple[list["OCRSample"], dict[str, int]]:
    from autocut.ocr import OCRSample, run_ocr_array_tasks_batched

    samples, timestamps, ocr_tasks, copy_sources, stats = _plan_ocr_frame_arrays(
        frames,
        sample_offset=0,
        min_score=min_score,
        signature_threshold=signature_threshold,
        min_white_pixels=min_white_pixels,
        rec_only=rec_only,
        cache_sources=cache_sources,
    )

    for index, sample in run_ocr_array_tasks_batched(
        ocr_tasks,
        workers=workers,
        ocr_threads=ocr_threads,
        engine_name=ocr_engine,
        model_size=ppocrv6_model_size,
        device=ocr_device,
        inference_engine=ocr_inference_engine,
        batch_size=ocr_batch_size,
    ):
        samples[index] = sample

    for index, source_index in copy_sources.items():
        source = samples[source_index] or OCRSample(timestamp=timestamps[source_index])
        samples[index] = OCRSample(
            timestamp=timestamps[index],
            text=source.text,
            score=source.score,
        )

    complete_samples = [
        sample if sample is not None else OCRSample(timestamp=timestamps[index])
        for index, sample in enumerate(samples)
    ]
    return complete_samples, stats


def _plan_ocr_frame_arrays(
    frames: list[tuple[float, object]],
    *,
    sample_offset: int,
    min_score: float,
    signature_threshold: int,
    min_white_pixels: int,
    rec_only: bool,
    cache_sources: dict[bytes, int] | None = None,
):
    from autocut.ocr import (
        MAX_SUBTITLE_CROP_HEIGHT,
        OCRSample,
        find_subtitle_bbox,
        signature_distance,
        subtitle_frame_signature_from_array,
    )

    samples: list[OCRSample | None] = [None] * len(frames)
    timestamps = [timestamp for timestamp, _ in frames]
    ocr_tasks = []
    copy_sources: dict[int, int] = {}
    stats = {"ocr_calls": 0, "skipped": 0, "reused": 0, "cached": 0}
    previous_signature = None
    previous_ocr_source: int | None = None

    for index, (timestamp, frame) in enumerate(frames):
        global_index = sample_offset + index
        full_signature = subtitle_frame_signature_from_array(
            frame,
            min_white_pixels=min_white_pixels,
        )
        if full_signature is None:
            samples[index] = OCRSample(timestamp=timestamp)
            previous_signature = None
            previous_ocr_source = None
            stats["skipped"] += 1
            continue

        mode = "full"
        task_frame = frame
        match_signature = full_signature
        if rec_only:
            bbox = find_subtitle_bbox(frame)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if 0 < crop.shape[0] <= MAX_SUBTITLE_CROP_HEIGHT and crop.shape[1] > 0:
                    task_frame = crop.copy()
                    mode = "rec_crop"
                    crop_signature = subtitle_frame_signature_from_array(
                        task_frame,
                        min_white_pixels=max(16, min_white_pixels // 4),
                    )
                    if crop_signature is not None:
                        match_signature = crop_signature
                else:
                    mode = "rec_frame"
            else:
                mode = "rec_frame"

        if cache_sources is not None and match_signature in cache_sources:
            copy_sources[global_index] = cache_sources[match_signature]
            previous_signature = match_signature
            previous_ocr_source = cache_sources[match_signature]
            stats["cached"] += 1
            continue

        distance = signature_distance(previous_signature, match_signature)
        if (
            distance is not None
            and distance <= signature_threshold
            and previous_ocr_source is not None
        ):
            copy_sources[global_index] = previous_ocr_source
            previous_signature = match_signature
            stats["reused"] += 1
            continue

        shape = task_frame.shape
        ocr_tasks.append((global_index, task_frame.tobytes(), shape, timestamp, min_score, mode))
        if cache_sources is not None:
            cache_sources[match_signature] = global_index
        previous_signature = match_signature
        previous_ocr_source = global_index
        stats["ocr_calls"] += 1

    return samples, timestamps, ocr_tasks, copy_sources, stats


@app.command()
def render(
    video: Path,
    timeline: Path,
    output: Path = typer.Option(Path("outputs/rough_cut.mp4"), "--output", "-o"),
    reencode: bool = typer.Option(
        True,
        "--reencode/--stream-copy",
        help="Re-encode for accurate cuts, or stream-copy for speed.",
    ),
) -> None:
    """Render a rough cut from a JSON timeline."""
    try:
        timeline_model = load_timeline(timeline)
        render_timeline(video, timeline_model.keep_segments(), output, reencode=reencode)
    except (FFmpegError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Rough cut written to {output}")


@app.command()
def transcribe(
    video: Path,
    output: Path = typer.Option(Path("runs/default/transcript.json"), "--output", "-o"),
    srt_output: Path | None = typer.Option(None, "--srt-output"),
    audio_output: Path = typer.Option(Path("runs/default/audio.wav"), "--audio-output"),
    sample_rate: int = typer.Option(16000, "--sample-rate"),
    model: str = typer.Option("paraformer-zh", "--model"),
    vad_model: str = typer.Option("fsmn-vad", "--vad-model"),
    punc_model: str = typer.Option("ct-punc", "--punc-model"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    """Transcribe a video into transcript JSON with FunASR."""
    from autocut.asr import ASRError, FunASRAdapter
    from autocut.transcript import save_transcript

    try:
        extract_audio(video, audio_output, sample_rate=sample_rate)
        adapter = FunASRAdapter(
            model=model,
            vad_model=vad_model,
            punc_model=punc_model,
            device=device,
        )
        transcript_model = adapter.transcribe(audio_output, source=video)
        save_transcript(transcript_model, output)
        if srt_output is not None:
            write_srt(transcript_model, srt_output)
    except (FFmpegError, ASRError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Transcript written to {output}")
    if srt_output is not None:
        typer.echo(f"SRT written to {srt_output}")


@app.command("llm-plan")
def llm_plan(
    job: str = typer.Option(..., "--job", "-j"),
    input_path: Path | None = typer.Option(None, "--input", "-i"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    model: str | None = typer.Option(None, "--model"),
    base_url: str | None = typer.Option(None, "--base-url"),
    api_key: str | None = typer.Option(None, "--api-key"),
    api_key_env: str = typer.Option("AUTOCUT_LLM_API_KEY", "--api-key-env"),
    wire_api: str | None = typer.Option(None, "--wire-api"),
    reasoning_effort: str | None = typer.Option(None, "--reasoning-effort"),
    temperature: float = typer.Option(0.2, "--temperature"),
    timeout: float = typer.Option(120, "--timeout"),
    json_mode: bool = typer.Option(True, "--json-mode/--no-json-mode"),
) -> None:
    """Generate runs/<job>/edit_plan.json from llm_input.md."""
    _generate_llm_plan(
        job=job,
        input_path=input_path,
        output=output,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        wire_api=wire_api,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        timeout=timeout,
        json_mode=json_mode,
    )


@app.command("decide-edits")
def decide_edits(
    job: str = typer.Option(..., "--job", "-j"),
    input_path: Path | None = typer.Option(None, "--input", "-i"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    model: str | None = typer.Option(None, "--model"),
    base_url: str | None = typer.Option(None, "--base-url"),
    api_key: str | None = typer.Option(None, "--api-key"),
    api_key_env: str = typer.Option("AUTOCUT_LLM_API_KEY", "--api-key-env"),
    wire_api: str | None = typer.Option(None, "--wire-api"),
    reasoning_effort: str | None = typer.Option(None, "--reasoning-effort"),
    temperature: float = typer.Option(0.2, "--temperature"),
    timeout: float = typer.Option(120, "--timeout"),
    json_mode: bool = typer.Option(True, "--json-mode/--no-json-mode"),
) -> None:
    """Alias for llm-plan."""
    _generate_llm_plan(
        job=job,
        input_path=input_path,
        output=output,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        wire_api=wire_api,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        timeout=timeout,
        json_mode=json_mode,
    )


def _generate_llm_plan(
    *,
    job: str,
    input_path: Path | None,
    output: Path | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    api_key_env: str,
    wire_api: str | None,
    reasoning_effort: str | None,
    temperature: float,
    timeout: float,
    json_mode: bool,
) -> None:
    from autocut.llm import LLMError, config_from_env, generate_edit_plan

    paths = _job_paths(job)
    llm_input = input_path or paths.llm_input
    edit_plan_output = output or paths.edit_plan

    if not llm_input.exists():
        typer.echo(f"LLM input file does not exist: {llm_input}")
        raise typer.Exit(1)

    try:
        config = config_from_env(
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            wire_api=wire_api,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            timeout=timeout,
            json_mode=json_mode,
        )
        edit_plan = generate_edit_plan(llm_input.read_text(encoding="utf-8"), config)
        edit_plan_output.parent.mkdir(parents=True, exist_ok=True)
        edit_plan_output.write_text(edit_plan.model_dump_json(indent=2), encoding="utf-8")
    except LLMError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Edit plan written to {edit_plan_output}")
    typer.echo(f"Next: python -m uv run autocut apply-plan --job {paths.job}")


@app.command("export-srt")
def export_srt(
    transcript: Path,
    output: Path = typer.Option(Path("outputs/subtitles.srt"), "--output", "-o"),
) -> None:
    """Export SRT subtitles from transcript JSON."""
    try:
        transcript_model = load_transcript(transcript)
        write_srt(transcript_model, output)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"SRT written to {output}")


@app.command("export-index-srt")
def export_index_srt(
    sentence_index: Path,
    output: Path = typer.Option(Path("outputs/sentences.srt"), "--output", "-o"),
) -> None:
    """Export SRT subtitles from sentence index JSON."""
    from autocut.semantic import load_sentence_index, write_sentence_index_srt

    try:
        sentence_index_model = load_sentence_index(sentence_index)
        write_sentence_index_srt(sentence_index_model, output)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"SRT written to {output}")


@app.command("build-index")
def build_index(
    transcript: Path,
    output: Path = typer.Option(Path("runs/default/sentence_index.json"), "--output", "-o"),
) -> None:
    """Build sentence-level index from transcript JSON."""
    from autocut.semantic import build_sentence_index, save_sentence_index

    try:
        transcript_model = load_transcript(transcript)
        sentence_index = build_sentence_index(transcript_model)
        save_sentence_index(sentence_index, output)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Sentence index written to {output}")


@app.command("build-blocks")
def build_blocks(
    sentence_index: Path,
    output: Path = typer.Option(Path("runs/default/topic_blocks.json"), "--output", "-o"),
    min_duration: float = typer.Option(20, "--min-duration"),
    max_duration: float = typer.Option(60, "--max-duration"),
    split_pause: float = typer.Option(1.2, "--split-pause"),
) -> None:
    """Build topic blocks from sentence index."""
    from autocut.semantic import build_topic_blocks, load_sentence_index, save_topic_blocks

    try:
        sentence_index_model = load_sentence_index(sentence_index)
        topic_blocks = build_topic_blocks(
            sentence_index_model,
            min_duration=min_duration,
            max_duration=max_duration,
            split_pause=split_pause,
        )
        save_topic_blocks(topic_blocks, output)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Topic blocks written to {output}")


@app.command("compact")
def compact(
    topic_blocks: Path,
    output: Path = typer.Option(Path("runs/default/llm_input.md"), "--output", "-o"),
) -> None:
    """Render topic blocks into compact LLM markdown input."""
    from autocut.semantic import load_topic_blocks, save_compact_markdown

    try:
        topic_blocks_model = load_topic_blocks(topic_blocks)
        save_compact_markdown(topic_blocks_model, output)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Compact LLM input written to {output}")


@app.command("assess-transcript")
def assess_transcript(
    sentence_index: Path,
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Assess sentence-index quality and recommend ASR vs subtitle/OCR."""
    from autocut.quality import assess_sentence_index, save_quality_report
    from autocut.semantic import load_sentence_index

    try:
        sentence_index_model = load_sentence_index(sentence_index)
        report = assess_sentence_index(sentence_index_model)
        if output is not None:
            save_quality_report(report, output)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(report.model_dump_json(indent=2))
    if output is not None:
        typer.echo(f"Quality report written to {output}")


@app.command("plan-to-timeline")
def plan_to_timeline(
    edit_plan: Path,
    topic_blocks: Path,
    output: Path = typer.Option(Path("runs/default/timeline.json"), "--output", "-o"),
) -> None:
    """Convert LLM edit_plan JSON into executable timeline JSON."""
    from autocut.semantic import edit_plan_to_timeline, load_edit_plan, load_topic_blocks

    try:
        edit_plan_model = load_edit_plan(edit_plan)
        topic_blocks_model = load_topic_blocks(topic_blocks)
        timeline = edit_plan_to_timeline(edit_plan_model, topic_blocks_model)
        save_timeline(timeline, output)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    typer.echo(f"Timeline written to {output}")
