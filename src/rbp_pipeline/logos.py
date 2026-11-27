"""Sequence logos with matplotlib only (no logomaker dependency)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from .attribution import information_content

ALPHABET = "ACGU"
COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "U": "#D62839"}
_FONT = FontProperties(family="DejaVu Sans", weight="bold")
_GLYPHS = {b: TextPath((0, 0), b, size=1, prop=_FONT) for b in ALPHABET}


def _letter(ax, letter: str, x: float, y: float, height: float, width: float = 0.9) -> None:
    if height <= 0:
        return
    path = _GLYPHS[letter]
    bb = path.get_extents()
    t = Affine2D().translate(-bb.x0, -bb.y0).scale(width / bb.width, height / bb.height).translate(x - width / 2, y)
    ax.add_patch(PathPatch(t.transform_path(path), facecolor=COLORS[letter], edgecolor="none"))


def draw_logo(ax, heights: np.ndarray, ylabel: str = "bits") -> None:
    """
    heights: (L, 4) letter heights in A, C, G, U order. Positive values stack upwards (smallest at the
    bottom), negative values stack downwards.
    """
    heights = np.asarray(heights, dtype=float)
    for i, row in enumerate(heights):
        up = 0.0
        for j in np.argsort(row):
            if row[j] > 0:
                _letter(ax, ALPHABET[j], i, up, row[j])
                up += row[j]
        down = 0.0
        for j in np.argsort(-row):
            if row[j] < 0:
                down += row[j]
                _letter(ax, ALPHABET[j], i, down, -row[j])
    lo, hi = min(heights.clip(max=0).sum(axis=1).min(), 0), max(heights.clip(min=0).sum(axis=1).max(), 0)
    pad = 0.05 * max(hi - lo, 1e-6)
    ax.set_xlim(-0.6, len(heights) - 0.4)
    ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if lo < 0:
        ax.axhline(0, color="black", lw=0.6)


def pwm_heights(pwm: np.ndarray) -> np.ndarray:
    """Information-content letter heights (bits) for a probability matrix."""
    pwm = np.asarray(pwm, dtype=float)
    return pwm * information_content(pwm)[:, None]
