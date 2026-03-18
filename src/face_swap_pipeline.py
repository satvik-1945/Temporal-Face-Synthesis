"""
Face swap pipeline for video: extract frames, swap faces, reconstruct video.

Uses InsightFace inswapper_128 for high-quality face swapping.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def ensure_inswapper_model() -> Path:
    """Download inswapper_128.onnx to ~/.insightface/models/ if not present."""
    model_dir = Path.home() / ".insightface" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "inswapper_128.onnx"

    if model_path.exists():
        return model_path

    try:
        from huggingface_hub import hf_hub_download

        # Download from Hugging Face (thebiglaskowski hosts it)
        downloaded = hf_hub_download(
            repo_id="thebiglaskowski/inswapper_128.onnx",
            filename="inswapper_128.onnx",
            local_dir=str(model_dir),
        )
        return Path(downloaded)
    except Exception as e:
        raise RuntimeError(
            f"Could not download inswapper_128.onnx. "
            f"Manually download from https://huggingface.co/thebiglaskowski/inswapper_128.onnx "
            f"and place in {model_dir}. Error: {e}"
        ) from e


def load_models(model_path: Optional[Path] = None):
    """Load FaceAnalysis (detection) and FaceSwapper models."""
    import insightface
    from insightface.app import FaceAnalysis

    model_path = model_path or ensure_inswapper_model()

    # Face detection + recognition (for embeddings)
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Face swapper
    swapper = insightface.model_zoo.get_model(
        str(model_path), download=False, download_zip=False
    )
    swapper.prepare(ctx_id=0)

    return app, swapper


def swap_face_in_image(
    app,
    swapper,
    source_img: np.ndarray,
    target_img: np.ndarray,
    source_face_index: int = 0,
    target_face_index: int = 0,
) -> np.ndarray:
    """
    Swap face from source image onto target image.

    Args:
        app: FaceAnalysis instance
        swapper: FaceSwapper model
        source_img: BGR image with the face to use (user's photo)
        target_img: BGR image with the face to replace (video frame)
        source_face_index: Which face in source (if multiple)
        target_face_index: Which face in target (if multiple)

    Returns:
        Target image with face swapped
    """
    source_faces = app.get(source_img)
    target_faces = app.get(target_img)

    if not source_faces:
        raise ValueError("No face detected in source image")
    if not target_faces:
        return target_img  # No face to swap, return original

    source_face = source_faces[min(source_face_index, len(source_faces) - 1)]
    target_face = target_faces[min(target_face_index, len(target_faces) - 1)]

    # swapper.get(img, target_face, source_face, paste_back=True)
    result = swapper.get(target_img, target_face, source_face, paste_back=True)
    return result


def process_video(
    video_path: str | Path,
    source_photo_path: str | Path,
    output_path: str | Path,
    *,
    max_frames: Optional[int] = None,
    fps_scale: float = 1.0,
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
        model_path: Optional path to inswapper model

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

    # Load models
    app, swapper = load_models(model_path)

    # Load source face image
    source_img = cv2.imread(str(source_photo_path))
    if source_img is None:
        raise ValueError(f"Could not read image: {source_photo_path}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
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
                    result = swap_face_in_image(app, swapper, source_img, frame)
                    out.write(result)
                except Exception as e:
                    # Fallback: write original frame if swap fails
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
