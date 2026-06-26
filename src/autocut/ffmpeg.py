import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from autocut.timeline import TimelineSegment
from autocut.video_info import VideoInfo


class FFmpegError(RuntimeError):
    """Raised when ffmpeg or ffprobe cannot complete a requested operation."""


@dataclass(frozen=True)
class FFmpegStatus:
    ok: bool
    missing: list[str]


@dataclass(frozen=True)
class SubtitleStream:
    index: int
    codec_name: str | None = None
    language: str | None = None
    title: str | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_tool(name: str) -> str | None:
    local_matches = sorted((_project_root() / "tools").glob(f"**/{name}.exe"))
    if local_matches:
        return str(local_matches[0])
    return shutil.which(name)


def ffmpeg_bin_dir() -> Path | None:
    ffmpeg_path = _find_tool("ffmpeg")
    if ffmpeg_path is None:
        return None
    return Path(ffmpeg_path).parent


def check_ffmpeg() -> FFmpegStatus:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if _find_tool(tool) is None]
    return FFmpegStatus(ok=not missing, missing=missing)


def _require_tools() -> None:
    status = check_ffmpeg()
    if not status.ok:
        missing = ", ".join(status.missing)
        raise FFmpegError(
            f"Missing required tool(s): {missing}. Install FFmpeg and add it to PATH."
        )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    executable = _find_tool(command[0])
    if executable is not None:
        command = [executable, *command[1:]]

    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise FFmpegError(
            "FFmpeg executable was not found. Install FFmpeg and add it to PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "No stderr output."
        raise FFmpegError(stderr) from exc


def _resolve_command(command: list[str]) -> list[str]:
    executable = _find_tool(command[0])
    if executable is not None:
        return [executable, *command[1:]]
    return command


def probe_video(video: Path) -> VideoInfo:
    _require_tools()
    if not video.exists():
        raise FFmpegError(f"Video file does not exist: {video}")

    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video),
        ]
    )
    payload = json.loads(result.stdout)
    return VideoInfo.from_ffprobe(payload)


def probe_subtitle_streams(video: Path) -> list[SubtitleStream]:
    _require_tools()
    if not video.exists():
        raise FFmpegError(f"Video file does not exist: {video}")

    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(video),
        ]
    )
    payload = json.loads(result.stdout)
    streams = []
    for stream in payload.get("streams", []):
        if stream.get("codec_type") != "subtitle":
            continue
        tags = stream.get("tags", {})
        streams.append(
            SubtitleStream(
                index=int(stream["index"]),
                codec_name=stream.get("codec_name"),
                language=tags.get("language"),
                title=tags.get("title"),
            )
        )
    return streams


def extract_audio(video: Path, output: Path, sample_rate: int = 16000) -> None:
    _require_tools()
    if not video.exists():
        raise FFmpegError(f"Video file does not exist: {video}")

    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            str(output),
        ]
    )


def extract_frame(
    video: Path,
    output: Path,
    timestamp: float,
    crop: str | None = None,
) -> None:
    _require_tools()
    if not video.exists():
        raise FFmpegError(f"Video file does not exist: {video}")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-update",
        "1",
    ]
    if crop:
        command.extend(["-vf", f"crop={crop}"])
    command.append(str(output))
    _run(command)


def extract_frames(
    video: Path,
    output_pattern: Path,
    *,
    fps: float,
    crop: str | None = None,
    start: float | None = None,
    duration: float | None = None,
) -> None:
    _require_tools()
    if not video.exists():
        raise FFmpegError(f"Video file does not exist: {video}")
    if fps <= 0:
        raise FFmpegError("FPS must be greater than 0.")

    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y"]
    if start is not None:
        command.extend(["-ss", f"{start:.3f}"])
    if duration is not None:
        command.extend(["-t", f"{duration:.3f}"])
    command.extend(["-i", str(video)])

    filters = [f"fps={fps}"]
    if crop:
        filters.append(f"crop={crop}")
    command.extend(["-vf", ",".join(filters), str(output_pattern)])
    _run(command)


def iter_frames(
    video: Path,
    *,
    fps: float,
    crop: str | None = None,
    start: float | None = None,
    duration: float | None = None,
):
    try:
        import numpy as np
    except ImportError as exc:
        raise FFmpegError("NumPy is required for in-memory frame streaming.") from exc

    _require_tools()
    if not video.exists():
        raise FFmpegError(f"Video file does not exist: {video}")
    if fps <= 0:
        raise FFmpegError("FPS must be greater than 0.")

    width, height = _frame_size(video, crop)
    command = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        command.extend(["-ss", f"{start:.3f}"])
    if duration is not None:
        command.extend(["-t", f"{duration:.3f}"])
    command.extend(["-i", str(video)])
    filters = [f"fps={fps}"]
    if crop:
        filters.append(f"crop={crop}")
    filters.append("format=bgr24")
    command.extend(["-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])

    process = subprocess.Popen(
        _resolve_command(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frame_size = width * height * 3
    try:
        assert process.stdout is not None
        while True:
            raw = process.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise FFmpegError("FFmpeg returned an incomplete raw frame.")
            yield np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()

        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise FFmpegError(stderr.strip() or "FFmpeg frame streaming failed.")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _frame_size(video: Path, crop: str | None) -> tuple[int, int]:
    if crop:
        match = re.match(r"^\s*(\d+):(\d+):", crop)
        if not match:
            raise FFmpegError("In-memory frame streaming requires numeric crop width and height.")
        return int(match.group(1)), int(match.group(2))

    info = probe_video(video)
    if info.width is None or info.height is None:
        raise FFmpegError("Could not determine video frame size.")
    return info.width, info.height


def render_timeline(
    video: Path,
    segments: list[TimelineSegment],
    output: Path,
    reencode: bool = True,
) -> None:
    _require_tools()
    if not video.exists():
        raise FFmpegError(f"Video file does not exist: {video}")
    if not segments:
        raise FFmpegError("Timeline has no keep segments to render.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="autocut_") as temp_root:
        temp_dir = Path(temp_root)
        clip_paths = []
        for index, segment in enumerate(segments, start=1):
            clip_path = temp_dir / f"clip_{index:04d}.mp4"
            _cut_segment(video, clip_path, segment, reencode=reencode)
            clip_paths.append(clip_path)

        concat_file = temp_dir / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in clip_paths),
            encoding="utf-8",
        )

        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output),
            ]
        )


def _cut_segment(
    video: Path,
    output: Path,
    segment: TimelineSegment,
    reencode: bool,
) -> None:
    base = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{segment.start:.3f}",
        "-to",
        f"{segment.end:.3f}",
        "-i",
        str(video),
        "-avoid_negative_ts",
        "make_zero",
    ]
    if reencode:
        command = [
            *base,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output),
        ]
    else:
        command = [*base, "-c", "copy", str(output)]
    _run(command)
