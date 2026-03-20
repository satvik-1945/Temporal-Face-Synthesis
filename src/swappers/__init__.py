"""
Face swapper backends: inswapper (128), ghost (256), simswap (256).
"""

from .base import SwapperBase
from .inswapper import InswapperSwapper
from .ghost import GhostSwapper
from .simswap import SimSwapSwapper

SWAPPERS = {
    "inswapper": InswapperSwapper,
    "ghost": GhostSwapper,
    "simswap": SimSwapSwapper,
}


def get_swapper(name: str) -> type[SwapperBase]:
    """Get swapper class by name."""
    if name not in SWAPPERS:
        raise ValueError(f"Unknown swapper: {name}. Choose from: {list(SWAPPERS)}")
    return SWAPPERS[name]
