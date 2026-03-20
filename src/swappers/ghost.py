"""Ghost 256 - higher quality, 256px resolution (ai-forever)."""

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import SwapperBase


def _match_color(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match src color stats to ref (LAB mean/std transfer). Removes grey box."""
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(3):
        s_mean, s_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std()
        r_mean, r_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std()
        if s_std > 1e-5 and r_std > 1e-5:
            src_lab[:, :, i] = (src_lab[:, :, i] - s_mean) * (r_std / s_std) + r_mean
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def _ensure_models(model_dir: Path) -> tuple[Path, Path]:
    """Download ghost_1_256.onnx and crossface_ghost.onnx if needed."""
    model_dir.mkdir(parents=True, exist_ok=True)
    ghost_path = model_dir / "ghost_1_256.onnx"
    crossface_path = model_dir / "crossface_ghost.onnx"

    try:
        from huggingface_hub import hf_hub_download

        if not ghost_path.exists():
            hf_hub_download(
                repo_id="facefusion/models-3.0.0",
                filename="ghost_1_256.onnx",
                local_dir=str(model_dir),
            )
        if not crossface_path.exists():
            hf_hub_download(
                repo_id="facefusion/models-3.4.0",
                filename="crossface_ghost.onnx",
                local_dir=str(model_dir),
            )
    except Exception as e:
        raise RuntimeError(
            f"Could not download Ghost models. Error: {e}"
        ) from e

    return ghost_path, crossface_path


class GhostSwapper(SwapperBase):
    name = "ghost"

    def __init__(self):
        self._size = (256, 256)
        self._mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def load(self, model_dir: Optional[Path] = None) -> tuple[Any, Any]:
        from insightface.app import FaceAnalysis
        import onnxruntime

        model_dir = model_dir or Path.home() / ".insightface" / "models"
        ghost_path, crossface_path = _ensure_models(model_dir)

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            face_swapper = onnxruntime.InferenceSession(
                str(ghost_path), providers=providers
            )
            embedding_converter = onnxruntime.InferenceSession(
                str(crossface_path), providers=providers
            )
        except Exception:
            face_swapper = onnxruntime.InferenceSession(
                str(ghost_path), providers=["CPUExecutionProvider"]
            )
            embedding_converter = onnxruntime.InferenceSession(
                str(crossface_path), providers=["CPUExecutionProvider"]
            )

        swapper = {
            "face_swapper": face_swapper,
            "embedding_converter": embedding_converter,
        }
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
        from insightface.utils import face_align

        source_faces = app.get(source_img)
        target_faces = app.get(target_img)
        if not source_faces:
            raise ValueError("No face detected in source image")
        if not target_faces:
            return target_img

        source_face = source_faces[min(source_face_index, len(source_faces) - 1)]
        target_face = target_faces[min(target_face_index, len(target_faces) - 1)]

        # Convert source embedding (buffalo_l 512 -> ghost format)
        emb = source_face.normed_embedding.reshape(-1, 512).astype(np.float32)
        conv = swapper["embedding_converter"]
        emb_converted = conv.run(None, {"input": emb})[0]
        emb_converted = emb_converted.ravel() / np.linalg.norm(emb_converted)
        source_embedding = emb_converted.reshape(1, -1).astype(np.float32)

        # Crop target face to 256x256
        aimg, M = face_align.norm_crop2(
            target_img, target_face.kps, self._size[0]
        )

        # Prepare target: BGR->RGB, /255, normalize
        target_blob = aimg[:, :, ::-1].astype(np.float32) / 255.0
        target_blob = (target_blob - self._mean) / self._std
        target_blob = target_blob.transpose(2, 0, 1)
        target_blob = np.expand_dims(target_blob, axis=0).astype(np.float32)

        # Forward
        face_swapper = swapper["face_swapper"]
        out = face_swapper.run(
            None,
            {"source": source_embedding, "target": target_blob},
        )[0][0]

        # Denormalize: (x * std + mean), clip, BGR
        out = out.transpose(1, 2, 0)
        out = out * self._std + self._mean
        out = np.clip(out, 0, 1)
        bgr_fake = (out[:, :, ::-1] * 255).astype(np.uint8)

        # Color transfer: match swapped face to target lighting (removes grey box)
        bgr_fake = _match_color(bgr_fake, aimg)

        # Paste back
        IM = cv2.invertAffineTransform(M)
        bgr_fake = cv2.warpAffine(
            bgr_fake, IM,
            (target_img.shape[1], target_img.shape[0]),
            borderValue=0.0,
        )

        # Feathered blend mask (erode + blur to avoid hard square edge)
        img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
        img_white = cv2.warpAffine(
            img_white, IM,
            (target_img.shape[1], target_img.shape[0]),
            borderValue=0.0,
        )
        img_white[img_white > 20] = 255
        img_mask = img_white.astype(np.uint8)
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)
        if len(mask_h_inds) > 0:
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h * mask_w))
            k_erode = max(mask_size // 10, 10)
            kernel = np.ones((k_erode, k_erode), np.uint8)
            img_mask = cv2.erode(img_mask, kernel, iterations=1)
            k_blur = max(mask_size // 20, 5)
            blur_size = (2 * k_blur + 1, 2 * k_blur + 1)
            img_mask = cv2.GaussianBlur(img_mask.astype(np.float32), blur_size, 0)
        mask = (img_mask / 255).reshape(*img_mask.shape, 1)
        result = (mask * bgr_fake + (1 - mask) * target_img.astype(np.float32)).astype(np.uint8)
        return result
