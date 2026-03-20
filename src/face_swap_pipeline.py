"""
Face swap pipeline for video: extract frames, swap faces, reconstruct video.

Supports multiple backends: inswapper (128px), ghost (256px), simswap (256px).
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_models(
    model_name: str = "inswapper",
    model_path: Optional[Path] = None,
) -> tuple[object, object, object]:
    """
    Load FaceAnalysis and face swapper.

    Args:
        model_name: "inswapper" (fast, 128px), "ghost" (256px), "simswap" (256px)
        model_path: Optional custom model directory

    Returns:
        (app, swapper, backend) - backend.swap(app, swapper, ...) does the swap
    """
    from src.swappers import get_swapper

    backend_class = get_swapper(model_name)
    backend = backend_class()
    app, swapper = backend.load(model_path)
    return app, swapper, backend


def swap_face_in_image(
    app: object,
    swapper: object,
    backend: object,
    source_img: np.ndarray,
    target_img: np.ndarray,
    source_face_index: int = 0,
    target_face_index: int = 0,
) -> np.ndarray:
    """
    Swap face from source image onto target image.

    Args:
        app: FaceAnalysis instance
        swapper: Model-specific swapper (INSwapper or dict for ghost/simswap)
        backend: SwapperBase instance (inswapper, ghost, simswap)
        source_img: BGR image with the face to use (user's photo)
        target_img: BGR image with the face to replace (video frame)
        source_face_index: Which face in source (if multiple)
        target_face_index: Which face in target (if multiple)

    Returns:
        Target image with face swapped
    """
    return backend.swap(
        app, swapper, source_img, target_img,
        source_face_index, target_face_index,
    )


def process_video(
    video_path: str | Path,
    source_photo_path: str | Path,
    output_path: str | Path,
    *,
    max_frames: Optional[int] = None,
    fps_scale: float = 1.0,
    model_name: str = "inswapper",
    model_path: Optional[Path] = None,
) -> Path:
    """
    Process video: swap face from source photo onto every frame.

    Args:
        video_path: Path to input video (template with face to replace)
        source_photo_path: Path to source face image (user's photo)
        output_path: Path for output video
        max_frames: Limit frames for testing (None = process all)
        fps_scale: Process every Nth frame (1.0 = all, 0.5 = every other)
        model_name: "inswapper", "ghost", or "simswap"
        model_path: Optional custom model directory

    Returns:
        Path to output video
    """
    video_path = Path(video_path)
    source_photo_path = Path(source_photo_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not source_photo_path.exists():
        raise FileNotFoundError(f"Source photo not found: {source_photo_path}")

    app, swapper, backend = load_models(model_name=model_name, model_path=model_path)

    source_img = cv2.imread(str(source_photo_path))
    if source_img is None:
        raise ValueError(f"Could not read image: {source_photo_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_interval = max(1, int(1.0 / fps_scale)) if fps_scale < 1.0 else 1
    frames_processed = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames is not None and frames_processed >= max_frames:
                break

            if frame_idx % frame_interval == 0:
                try:
                    result = swap_face_in_image(
                        app, swapper, backend, source_img, frame
                    )
                    out.write(result)
                except Exception as e:
                    print(f"Warning: frame {frame_idx} failed ({e}), using original")
                    out.write(frame)
                frames_processed += 1
            else:
                out.write(frame)

            frame_idx += 1

    finally:
        cap.release()
        out.release()

    print(f"Processed {frames_processed} frames, output: {output_path}")
    return output_path
