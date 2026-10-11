"""Video preprocessing utilities."""

from contextlib import contextmanager
import math
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from .image_preprocessing import resize_image

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _get_processed_frame(
    frame: np.ndarray,
    target_size: tuple[int, int] | list[int],
) -> np.ndarray:
    """Convert a BGR frame to RGB and resize it."""
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return resize_image(frame_rgb, target_size)


def _get_sampling_indices(
    video_num_frames: int,
    target_num_frames: int,
) -> np.ndarray:
    """Return approximately uniformly spaced frame indices."""
    if video_num_frames <= 0:
        raise ValueError("video_num_frames must be positive")
    if target_num_frames <= 0:
        raise ValueError("target_num_frames must be positive")
    indices = (
        (np.arange(target_num_frames) + 0.5)
        * (video_num_frames / target_num_frames)
    ).astype(int)
    return np.minimum(indices, video_num_frames - 1)


@contextmanager
def video_capture(filepath):
    """Yield a video capture and release it when the context exits."""
    cap = cv.VideoCapture(filepath)
    try:
        yield cap
    finally:
        cap.release()


def preprocess_videos(
    video_path: str | Path,
    target_num_frames: int,
    target_size: tuple[int, int] | list[int],
    dtype: torch.dtype = torch.float16,
    dense_sampling: bool = False,
) -> tuple[list[Path], list[torch.Tensor]]:
    """Load and preprocess a video file or the videos in a directory.

    Directories are scanned in filename order without
    visiting subdirectories. Only files with extensions in
    VIDEO_EXTENSIONS are processed.

    Args:
        video_path: Path to a video file or a directory
            containing videos.
        target_num_frames: Number of frames per sampled clip.
        target_size: Output frame size as (width, height).
        dtype: Data type of the output tensors.
        dense_sampling: Sample one frame per second and
            group frames into clips if True; otherwise sample one clip
            uniformly across each video. Videos shorter than
            target_num_frames seconds use uniform sampling
            in either mode.

    Returns:
        Video paths and their corresponding preprocessed tensors,
        in the same order. Each tensor has shape (B, L, C, H, W),
        where B is the number of clips if dense_sampling is True,
        otherwise it is 1, L is target_num_frames, and C is 3.
        Tensors are placed on the default CUDA device.
    """
    if target_num_frames <= 0:
        raise ValueError("target_num_frames must be positive")
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"{path} does not exist.")

    if dense_sampling:
        video_to_frames_array = video_to_dense_frames_array
    else:
        video_to_frames_array = video_to_sparse_frames_array

    if path.is_dir():
        videos_path = [
            filepath
            for filepath in sorted(path.iterdir(), key=lambda p: p.name)
            if filepath.is_file()
            and filepath.suffix.lower() in VIDEO_EXTENSIONS
        ]
    elif path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        videos_path = [path]
    else:
        raise RuntimeError(
            f"{path} is not a directory nor a video file."
        )
    
    videos_tensor = []
    for filepath in videos_path:
        video_np = video_to_frames_array(
            filepath,
            target_num_frames=target_num_frames,
            target_size=target_size,
        )
        videos_tensor.append(
            video_np_to_tensor(video_np, dtype=dtype)
        )
    
    return videos_path, videos_tensor


def video_to_sparse_frames_array(
    filepath: str | Path,
    *,
    target_num_frames: int,
    target_size: tuple[int, int] | list[int],
) -> np.ndarray:
    """Load approximately uniformly spaced RGB frames from a video.

    Resize each sampled frame to the target size. If the frame count
    is unavailable, read the entire video before selecting frames.

    Args:
        filepath: Path to the video file.
        target_num_frames: Number of frames to sample.
        target_size: Output frame size as (width, height).

    Returns:
        An array of RGB frames with shape
        (1, target_num_frames, H, W, 3), containing pixel values
        before normalization.
    """
    if target_num_frames <= 0:
        raise ValueError("target_num_frames must be positive")
    selected_frames = []
    with video_capture(filepath) as cap:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {filepath}")

        video_num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if video_num_frames <= 0:
            frames = []
            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    frames.append(frame)
                else:
                    break
            if not frames:
                raise RuntimeError(f"No readable frames in {filepath}")
            indices = _get_sampling_indices(len(frames), target_num_frames)
            for idx in indices:
                selected_frames.append(
                    _get_processed_frame(frames[idx], target_size)
                )

        else:
            indices = _get_sampling_indices(
                video_num_frames, 
                target_num_frames,
            )
            for idx in indices:
                if not cap.set(cv.CAP_PROP_POS_FRAMES, int(idx)):
                    raise RuntimeError(f"Couldn't seek to frame {int(idx)}")

                success, frame = cap.read()
                if not success:
                    raise RuntimeError(f"Cannot read frame {int(idx)}")

                selected_frames.append(
                    _get_processed_frame(frame, target_size)
                )
    frames_np = np.stack(selected_frames, axis=0)
    return np.expand_dims(frames_np, axis=0)


def video_to_dense_frames_array(
    filepath: str | Path,
    *,
    target_num_frames: int,
    target_size: tuple[int, int] | list[int],
) -> np.ndarray:
    """Sample RGB frames once per second and group them into clips.

    Sample at timestamps starting from zero and resize each frame to
    the target size. Discard trailing frames that do not form a
    complete clip. Use uniform sampling to return one clip when
    the video duration in seconds is less than target_num_frames.

    Args:
        filepath: Path to the video file.
        target_num_frames: Number of sampled frames per clip.
        target_size: Output frame size as (width, height).

    Returns:
        An array of RGB frames with shape
        (B, target_num_frames, H, W, 3), containing pixel values before
        normalization. B is the number of complete clips.
    """
    if target_num_frames <= 0:
        raise ValueError("target_num_frames must be positive")
    selected_frames = []
    with video_capture(filepath) as cap:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {filepath}")

        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)

        if (
            not math.isfinite(fps)
            or not math.isfinite(frame_count)
            or fps <= 0
            or frame_count <= 0
        ):
            raise RuntimeError("Cannot determine video duration")

        duration = frame_count / fps

        if duration < target_num_frames:
            return video_to_sparse_frames_array(
                filepath,
                target_num_frames=target_num_frames,
                target_size=target_size,
            )

        for timestamp in range(math.ceil(duration)):
            if not cap.set(cv.CAP_PROP_POS_MSEC, timestamp * 1000):
                raise RuntimeError(f"Couldn't seek to frame {timestamp}")

            success, frame = cap.read()
            if not success:
                break

            selected_frames.append(
                _get_processed_frame(frame, target_size)
            )

    num_frames_to_drop = len(selected_frames) % target_num_frames
    if num_frames_to_drop:
        selected_frames = selected_frames[:-num_frames_to_drop]
    if not selected_frames:
        raise RuntimeError(
            f"Not enough readable frames for a complete clip in {filepath}"
        )
    frames_np = np.stack(selected_frames, axis=0)
    B = len(selected_frames) // target_num_frames
    _, H, W, C = frames_np.shape
    return frames_np.reshape(B, target_num_frames, H, W, C)


def normalize_array(data: np.ndarray) -> np.ndarray:
    """Normalize RGB pixel values using ImageNet statistics.

    Expect values in [0, 255] with the channel dimension last.
    """
    return (data / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def video_np_to_tensor(
    video_np: np.ndarray,
    dtype: torch.dtype,
    device: torch.device | str = torch.device("cuda"),
) -> torch.Tensor:
    """Normalize RGB video clips and convert them to a PyTorch tensor.

    Args:
        video_np: Array of RGB pixel values in [0, 255], with shape
            (B, L, H, W, C).
        dtype: Data type of the output tensor.
        device: Device on which to place the output tensor.

    Returns:
        An ImageNet-normalized tensor with shape (B, L, C, H, W), using
        the requested data type and device.
    """
    video_np = normalize_array(video_np)
    # (B, L, H, W, C) -> (B, L, C, H, W)
    video_np = np.transpose(video_np, (0, 1, 4, 2, 3))

    video_tensor = torch.from_numpy(video_np).to(
        device=device, dtype=dtype, non_blocking=True
    )
    return video_tensor