"""
color_engine.py
───────────────
Dominant color palette extraction in CIELAB, fixed-size color
histogram for Milvus ANN, and Earth Mover's Distance scoring
for perceptually accurate re-ranking.
"""

import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color as skcolor

try:
    from pyemd import emd
    _HAS_EMD = True
except ImportError:
    _HAS_EMD = False

COLOR_HIST_DIM = 48          # 16 bins x 3 channels (L, a, b)
COLOR_HIST_BINS = 16
N_PALETTE_COLORS = 5
COLOR_TAU = 50.0             # normalization constant for EMD


# =====================================================
#        DOMINANT COLOR PALETTE (CIELAB)
# =====================================================

def extract_color_palette(
    fg_image: Image.Image,
    n_colors: int = N_PALETTE_COLORS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means in CIELAB on the foreground image.

    Returns:
        centers     – (n_colors, 3) LAB coordinates, sorted by proportion
        proportions – (n_colors,)   fraction of pixels per cluster
    """
    img_rgb = np.array(fg_image.resize((64, 64))).astype(np.float64) / 255.0
    img_lab = skcolor.rgb2lab(img_rgb)
    pixels = img_lab.reshape(-1, 3)

    chroma = np.sqrt(pixels[:, 1] ** 2 + pixels[:, 2] ** 2)
    mask = chroma > 5
    pixels = pixels[mask] if mask.sum() > n_colors else pixels

    km = KMeans(n_clusters=n_colors, n_init=3, random_state=0)
    km.fit(pixels)
    centers = km.cluster_centers_
    proportions = np.bincount(km.labels_, minlength=n_colors).astype(float)
    proportions /= proportions.sum() + 1e-10

    order = np.argsort(-proportions)
    return centers[order], proportions[order]


def palette_to_json(centers: np.ndarray, proportions: np.ndarray) -> str:
    """Serialize palette to JSON for storage in Milvus VARCHAR field."""
    data = {
        "centers": centers.tolist(),
        "proportions": proportions.tolist(),
    }
    return json.dumps(data)


def palette_from_json(s: str) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(s)
    return np.array(data["centers"]), np.array(data["proportions"])


# =====================================================
#        COLOR HISTOGRAM (48-dim, for Milvus ANN)
# =====================================================

def compute_color_histogram(
    fg_image: Image.Image,
    bins: int = COLOR_HIST_BINS,
) -> np.ndarray:
    """
    Fixed-size LAB histogram vector (48-dim), L1-normalized.
    Suitable for ANN search in Milvus.
    """
    img_rgb = np.array(fg_image.resize((64, 64))).astype(np.float64) / 255.0
    img_lab = skcolor.rgb2lab(img_rgb)

    h_l = np.histogram(img_lab[:, :, 0].ravel(), bins=bins, range=(0, 100))[0]
    h_a = np.histogram(img_lab[:, :, 1].ravel(), bins=bins, range=(-128, 127))[0]
    h_b = np.histogram(img_lab[:, :, 2].ravel(), bins=bins, range=(-128, 127))[0]

    hist = np.concatenate([h_l, h_a, h_b]).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# =====================================================
#        COLOR SIMILARITY (EMD for re-ranking)
# =====================================================

def _lab_distance_matrix(
    centers_a: np.ndarray,
    centers_b: np.ndarray,
) -> np.ndarray:
    """Euclidean distance in CIELAB between every pair of palette centers."""
    n = len(centers_a)
    m = len(centers_b)
    dist = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            dist[i, j] = np.linalg.norm(centers_a[i] - centers_b[j])
    return dist


def color_similarity_emd(
    palette_json_q: str,
    palette_json_r: str,
) -> float:
    """
    Earth Mover's Distance between two palettes, normalized to [0, 1].

    S_color = 1 - EMD / tau, clamped to [0, 1].

    Falls back to histogram cosine similarity if pyemd is unavailable.
    """
    c_q, p_q = palette_from_json(palette_json_q)
    c_r, p_r = palette_from_json(palette_json_r)

    if not _HAS_EMD:
        v_q = p_q / (np.linalg.norm(p_q) + 1e-10)
        v_r = p_r / (np.linalg.norm(p_r) + 1e-10)
        return float(np.clip(np.dot(v_q, v_r), 0.0, 1.0))

    dist_matrix = _lab_distance_matrix(c_q, c_r)

    p_q_64 = p_q.astype(np.float64)
    p_r_64 = p_r.astype(np.float64)
    p_q_64 /= p_q_64.sum()
    p_r_64 /= p_r_64.sum()

    emd_val = emd(p_q_64, p_r_64, dist_matrix)
    score = 1.0 - emd_val / COLOR_TAU
    return float(np.clip(score, 0.0, 1.0))


def color_similarity_hist(hist_q: np.ndarray, hist_r: np.ndarray) -> float:
    """Fast cosine similarity on color histograms (for candidate generation)."""
    dot = np.dot(hist_q, hist_r)
    norm = np.linalg.norm(hist_q) * np.linalg.norm(hist_r) + 1e-10
    return float(dot / norm)
