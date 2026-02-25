"""
shape_engine.py
───────────────
Edge-based shape descriptors: Hu Moments (7-dim, rotation/scale/translation
invariant) and contour angle histogram (64-dim, for Milvus ANN search).
"""

import cv2
import numpy as np
from PIL import Image

SHAPE_HIST_DIM = 64
HU_DIM = 7


def extract_shape_features(
    fg_image: Image.Image,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Hu Moments and contour angle histogram from a preprocessed
    foreground image.

    Returns:
        hu_moments      – (7,)  log-scale Hu Moments
        shape_histogram – (64,) L1-normalized contour angle histogram
    """
    gray = cv2.cvtColor(np.array(fg_image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    moments = cv2.moments(edges)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    angles: list[float] = []
    for c in contours:
        if len(c) < 3:
            continue
        diffs = np.diff(c.squeeze(1), axis=0).astype(float)
        a = np.arctan2(diffs[:, 1], diffs[:, 0])
        angles.extend(a.tolist())

    hist = np.histogram(
        angles, bins=SHAPE_HIST_DIM, range=(-np.pi, np.pi)
    )[0].astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

    return hu.astype(np.float32), hist


def shape_similarity(hu_q: np.ndarray, hu_r: np.ndarray) -> float:
    """Cosine similarity between two Hu Moment vectors."""
    hu_q = np.asarray(hu_q, dtype=np.float64)
    hu_r = np.asarray(hu_r, dtype=np.float64)
    dot = np.dot(hu_q, hu_r)
    norm = np.linalg.norm(hu_q) * np.linalg.norm(hu_r) + 1e-10
    return float(dot / norm)
