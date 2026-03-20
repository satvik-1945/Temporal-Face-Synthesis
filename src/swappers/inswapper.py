"""InsightFace INSwapper 128 - fast, 128px resolution."""

from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import SwapperBase


def _ensure_model(model_dir: Path) -> Path:
    model_path = model_dir / "inswapper_128.onnx"
    if model_path.exists():
        return model_path
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="thebiglaskowski/inswapper_128.onnx",
            filename="inswapper_128.onnx",
            local_dir=str(model_dir),
        )
        return Path(downloaded)
    except Exception as e:
        raise RuntimeError(
            f"Could not download inswapper_128.onnx. "
            f"Manually download and place in {model_dir}. Error: {e}"
        ) from e


class InswapperSwapper(SwapperBase):
    name = "inswapper"

    def load(self, model_dir: Optional[Path] = None) -> tuple[Any, Any]:
        from insightface.app import FaceAnalysis
        import insightface

        model_dir = model_dir or Path.home() / ".insightface" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = _ensure_model(model_dir)

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

        swapper = insightface.model_zoo.get_model(
            str(model_path), download=False, download_zip=False
        )
        return app, swapper

    def swap(
        self,
        app: Any,
        swapper: Any,
        source_img: np.ndarray,
        target_img: np.ndarray,
        source_face_index: int = 0,
        target_face_index: int = 0,
    ) -> np.ndarray:
        source_faces = app.get(source_img)
        target_faces = app.get(target_img)
        if not source_faces:
            raise ValueError("No face detected in source image")
        if not target_faces:
            return target_img

        source_face = source_faces[min(source_face_index, len(source_faces) - 1)]
        target_face = target_faces[min(target_face_index, len(target_faces) - 1)]
        return swapper.get(target_img, target_face, source_face, paste_back=True)
