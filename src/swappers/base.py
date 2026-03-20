"""Base interface for face swappers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np


class SwapperBase(ABC):
    """Abstract base for face swapper backends."""

    name: str = "base"

    @abstractmethod
    def load(self, model_dir: Optional[Path] = None) -> Any:
        """Load models. Returns (app, swapper) where app is FaceAnalysis."""
        ...

    @abstractmethod
    def swap(
        self,
        app: Any,
        swapper: Any,
        source_img: np.ndarray,
        target_img: np.ndarray,
        source_face_index: int = 0,
        target_face_index: int = 0,
    ) -> np.ndarray:
        """Swap face from source onto target. Returns BGR image."""
        ...
