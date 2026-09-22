"""Analysis and visualisation of US Federal Reserve rate decisions."""

from .dataset import (
    COLUMNS,
    build_frame,
    frame_path,
    load_frame,
    recession_spans,
    save_frame,
    tidy,
)
from .theme import CATEGORICAL, PALETTE, economist, register, save

__version__ = "0.1.0"
__all__ = [
    "CATEGORICAL",
    "COLUMNS",
    "PALETTE",
    "__version__",
    "build_frame",
    "economist",
    "frame_path",
    "load_frame",
    "recession_spans",
    "register",
    "save_frame",
    "save",
    "tidy",
]
