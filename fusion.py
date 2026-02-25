"""
fusion.py
─────────
Score normalization, adaptive weight selection, Reciprocal Rank
Fusion (RRF), and combined final-score computation across all
modalities: DINO, VGG, text, color, font, shape.

Includes pHash near-duplicate short-circuit, perfect text match
override, and soft text override for brand name matching.
"""

import numpy as np
from dataclasses import dataclass, field


# ── modality keys ─────────────────────────────────────────────────────────

MODALITIES = ("dino", "vgg", "text", "color", "font", "shape")

# ── adaptive weight profiles ──────────────────────────────────────────────
#    (dino, vgg, text, color, font, shape)

WEIGHTS_BOTH_TEXT = {
    "dino": 0.20, "vgg": 0.05, "text": 0.25,
    "color": 0.10, "font": 0.20, "shape": 0.20,
}
WEIGHTS_NO_TEXT = {
    "dino": 0.35, "vgg": 0.10, "text": 0.00,
    "color": 0.20, "font": 0.00, "shape": 0.35,
}
WEIGHTS_ONE_TEXT = {
    "dino": 0.25, "vgg": 0.10, "text": 0.15,
    "color": 0.15, "font": 0.10, "shape": 0.25,
}

RRF_K = 60
LAMBDA = 0.5


# =====================================================
#              SCORE NORMALIZATION
# =====================================================

def z_normalize(scores: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1-D array of scores. Returns zeros if std ≈ 0."""
    mu = scores.mean()
    sigma = scores.std()
    if sigma < 1e-8:
        return np.zeros_like(scores)
    return (scores - mu) / sigma


# =====================================================
#              ADAPTIVE WEIGHT SELECTION
# =====================================================

def select_weights(q_has_text: bool, c_has_text: bool) -> dict[str, float]:
    if q_has_text and c_has_text:
        return WEIGHTS_BOTH_TEXT
    elif not q_has_text and not c_has_text:
        return WEIGHTS_NO_TEXT
    else:
        return WEIGHTS_ONE_TEXT


# =====================================================
#              RECIPROCAL RANK FUSION
# =====================================================

def _ranks(scores: np.ndarray) -> np.ndarray:
    """Convert scores to 1-based ranks (highest score → rank 1)."""
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


# =====================================================
#             CANDIDATE DATA STRUCTURE
# =====================================================

@dataclass
class CandidateScores:
    """Raw per-modality scores for a set of candidates."""
    ids: list
    filenames: list[str]
    dino: np.ndarray
    vgg: np.ndarray
    text: np.ndarray
    color: np.ndarray
    font: np.ndarray
    shape: np.ndarray
    has_text: list[bool]
    phash_boost: np.ndarray | None = None


# =====================================================
#              FULL RE-RANKING PIPELINE
# =====================================================

def rerank(
    candidates: CandidateScores,
    q_has_text: bool,
    top_k: int = 20,
) -> list[dict]:
    """
    Re-rank candidates using adaptive weights, z-score normalization,
    and Reciprocal Rank Fusion.

    Short-circuits:
      - pHash near-duplicate: phash_boost > 0.95 → final = max(fusion, phash)
      - Perfect text match:   text_score >= 0.99  → final = 1.0
      - Soft text override:   text_score >= 0.85  → final = max(fusion, text)

    Returns a sorted list of dicts:
      {id, filename, final_score, dino_score, vgg_score, text_score,
       color_score, font_score, shape_score}
    """
    n = len(candidates.ids)
    if n == 0:
        return []

    raw = {
        "dino":  candidates.dino,
        "vgg":   candidates.vgg,
        "text":  candidates.text,
        "color": candidates.color,
        "font":  candidates.font,
        "shape": candidates.shape,
    }

    z_scores = {m: z_normalize(raw[m]) for m in MODALITIES}
    mod_ranks = {m: _ranks(raw[m]) for m in MODALITIES}

    final_scores = np.zeros(n)

    for i in range(n):
        # pHash near-duplicate short-circuit
        if (
            candidates.phash_boost is not None
            and candidates.phash_boost[i] > 0.95
        ):
            final_scores[i] = candidates.phash_boost[i]
            continue

        # Perfect text match short-circuit
        if q_has_text and candidates.has_text[i] and raw["text"][i] >= 0.99:
            final_scores[i] = 1.0
            continue

        c_has_text = candidates.has_text[i]
        w = select_weights(q_has_text, c_has_text)

        rrf = 0.0
        weighted_z = 0.0
        for m in MODALITIES:
            wm = w[m]
            if wm <= 0:
                continue
            rrf += wm / (RRF_K + mod_ranks[m][i])
            weighted_z += wm * z_scores[m][i]

        final_scores[i] = LAMBDA * rrf + (1.0 - LAMBDA) * weighted_z

        # Soft text override: brand name matches dominate
        if q_has_text and candidates.has_text[i] and raw["text"][i] >= 0.85:
            final_scores[i] = max(final_scores[i], raw["text"][i])

    order = np.argsort(-final_scores)
    results = []
    for idx in order[:top_k]:
        results.append({
            "id": candidates.ids[idx],
            "filename": candidates.filenames[idx],
            "final_score": round(
                float(np.clip(final_scores[idx], 0.0, 1.0)) * 100, 2
            ),
            "dino_score": round(float(raw["dino"][idx]), 3),
            "vgg_score": round(float(raw["vgg"][idx]), 3),
            "text_score": round(float(raw["text"][idx]), 3),
            "color_score": round(float(raw["color"][idx]), 3),
            "font_score": round(float(raw["font"][idx]), 3),
            "shape_score": round(float(raw["shape"][idx]), 3),
        })

    return results
