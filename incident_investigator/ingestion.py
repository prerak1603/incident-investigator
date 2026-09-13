from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_FPS = 25.0
MAX_PLAUSIBLE_FPS = 240.0  # some corrupt containers report absurd fps instead of 0


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray


@dataclass
class Window:
    start: float
    end: float
    frames: list[Frame]


@dataclass
class IngestResult:
    video_path: str
    fps: float
    duration: float
    frame_count: int
    windows: list[Window]
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def _sample_frames(cap: cv2.VideoCapture, fps: float, target_fps: float) -> list[Frame]:
    # reading sequentially and picking frames off elapsed time is more robust than
    # seeking with CAP_PROP_POS_FRAMES, which drifts on variable-frame-rate footage
    frames = []
    min_gap = 1.0 / target_fps
    next_sample_time = 0.0
    frame_idx = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        timestamp = frame_idx / fps
        if timestamp + 1e-6 >= next_sample_time:
            frames.append(Frame(index=frame_idx, timestamp=timestamp, image=image))
            next_sample_time += min_gap
        frame_idx += 1
    return frames


def _make_windows(frames: list[Frame], duration: float, window_size: float, stride: float) -> list[Window]:
    if not frames:
        return []
    if duration <= window_size:
        return [Window(start=0.0, end=duration, frames=frames)]

    windows = []
    start = 0.0
    while start + window_size <= duration + 1e-6:
        end = start + window_size
        window_frames = [f for f in frames if start - 1e-6 <= f.timestamp < end - 1e-6]
        if window_frames:
            windows.append(Window(start=start, end=end, frames=window_frames))
        start += stride
    return windows


def ingest_video(
    path: str,
    target_fps: float = 2.0,
    window_size: float = 2.0,
    stride: float = 1.0,
) -> IngestResult:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return IngestResult(path, fps=0.0, duration=0.0, frame_count=0, windows=[],
                             error=f"could not open video: {path}")

    warnings = []
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0 or native_fps > MAX_PLAUSIBLE_FPS:
        warnings.append(f"invalid fps reported ({native_fps}), falling back to {DEFAULT_FPS}")
        native_fps = DEFAULT_FPS

    frames = _sample_frames(cap, native_fps, target_fps)
    cap.release()

    if not frames:
        return IngestResult(path, fps=native_fps, duration=0.0, frame_count=0, windows=[],
                             warnings=warnings, error="no frames could be read")

    duration = frames[-1].timestamp + (1.0 / native_fps)  # last frame's own span, not just its start
    if duration < window_size:
        warnings.append(f"video duration ({duration:.2f}s) shorter than window size ({window_size}s); using a single window")

    windows = _make_windows(frames, duration, window_size, stride)
    return IngestResult(path, fps=native_fps, duration=duration, frame_count=len(frames),
                         windows=windows, warnings=warnings)
