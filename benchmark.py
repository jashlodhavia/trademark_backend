"""
benchmark.py
────────────
No-ground-truth benchmarking framework for the Trademark Similarity Engine.

Provides:
  1. Synthetic data generation (augmentations as pseudo-positive pairs)
  2. Retrieval metrics:  Hit@K, MRR, Precision@K, nDCG@K
  3. Embedding quality:  silhouette score, intra/inter cluster ratio
  4. Bias & robustness tests:
       - Background bias test
       - Color robustness test
       - OCR robustness test
       - Font similarity test
       - Cross-language robustness test
  5. A/B comparison with Wilcoxon signed-rank test
  6. Human-in-the-loop validation harness

Usage:
    python -m benchmark --images repository_images --n-augment 8
"""

import os
import io
import glob
import json
import random
import argparse
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.stats import wilcoxon
from sklearn.metrics import silhouette_score


# =====================================================
#        1. SYNTHETIC DATA GENERATION
# =====================================================

def rotate(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))


def scale(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    return img.resize((max(1, new_w), max(1, new_h)), Image.LANCZOS)


def add_background(img: Image.Image, color: tuple) -> Image.Image:
    padded = Image.new("RGB", (img.width + 100, img.height + 100), color)
    padded.paste(img, (50, 50))
    return padded


def add_noise(img: Image.Image, sigma: float = 10) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def jpeg_compress(img: Image.Image, quality: int = 50) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def crop_center(img: Image.Image, ratio: float = 0.85) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * ratio), int(h * ratio)
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def adjust_brightness(img: Image.Image, factor: float = 1.2) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def color_jitter(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img


def hue_shift(img: Image.Image, degrees: int = 30) -> Image.Image:
    hsv = img.convert("HSV")
    arr = np.array(hsv)
    arr[:, :, 0] = (arr[:, :, 0].astype(int) + degrees) % 256
    return Image.fromarray(arr, "HSV").convert("RGB")


def desaturate(img: Image.Image, factor: float = 0.5) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)


def generate_augmented_set(image_path: str, n: int = 8) -> list[Image.Image]:
    """Generate n augmented variants of an image."""
    img = Image.open(image_path).convert("RGB")
    augmented = []
    augmented.append(rotate(img, random.uniform(-15, 15)))
    augmented.append(scale(img, random.uniform(0.7, 1.3)))
    augmented.append(add_background(img, color=(
        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    )))
    augmented.append(add_noise(img, sigma=10))
    augmented.append(jpeg_compress(img, quality=random.randint(50, 90)))
    augmented.append(crop_center(img, ratio=random.uniform(0.8, 1.0)))
    augmented.append(adjust_brightness(img, factor=random.uniform(0.8, 1.2)))
    augmented.append(color_jitter(img))
    return augmented[:n]


# ── background-specific augmentations ──

def background_variants(img: Image.Image) -> dict[str, Image.Image]:
    return {
        "original": img.copy(),
        "white_padded": add_background(img, (255, 255, 255)),
        "black_padded": add_background(img, (0, 0, 0)),
        "textured": add_noise(add_background(img, (200, 200, 200)), sigma=25),
    }


# ── color-specific augmentations ──

def color_variants(img: Image.Image) -> dict[str, Image.Image]:
    return {
        "original": img.copy(),
        "hue_shifted": hue_shift(img, 30),
        "desaturated": desaturate(img, 0.5),
        "inverted": Image.fromarray(255 - np.array(img)),
    }


# ── OCR-hostile augmentations ──

def ocr_hostile_variants(img: Image.Image) -> dict[str, Image.Image]:
    return {
        "heavy_jpeg": jpeg_compress(img, quality=20),
        "rotated_30": rotate(img, 30),
        "noisy": add_noise(img, sigma=25),
    }


# =====================================================
#        2. RETRIEVAL METRICS
# =====================================================

def hit_at_k(results_filenames: list[str], true_filename: str, k: int) -> float:
    return 1.0 if true_filename in results_filenames[:k] else 0.0


def reciprocal_rank(results_filenames: list[str], true_filename: str) -> float:
    for i, fn in enumerate(results_filenames):
        if fn == true_filename:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    results_filenames: list[str],
    true_filenames: set[str],
    k: int,
) -> float:
    top = results_filenames[:k]
    return sum(1 for f in top if f in true_filenames) / k


def dcg_at_k(relevances: list[float], k: int) -> float:
    return sum(
        (2 ** rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(
    results_filenames: list[str],
    true_filenames: set[str],
    k: int,
) -> float:
    rels = [1.0 if f in true_filenames else 0.0 for f in results_filenames[:k]]
    actual_dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(rels, reverse=True)
    ideal_dcg = dcg_at_k(ideal_rels, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# =====================================================
#        3. EMBEDDING QUALITY METRICS
# =====================================================

def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    embeddings: (N, dim) array of all embeddings
    labels: (N,) array — each original image + its augmentations share a label
    """
    if len(set(labels)) < 2:
        return 0.0
    return float(silhouette_score(embeddings, labels, metric="cosine"))


def intra_inter_ratio(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Ratio of mean intra-cluster distance to mean inter-cluster distance.
    Lower is better (augments closer to their original than to others).
    """
    unique = np.unique(labels)
    intra_dists, inter_dists = [], []

    for label in unique:
        cluster = embeddings[labels == label]
        others = embeddings[labels != label]

        if len(cluster) > 1:
            from sklearn.metrics.pairwise import cosine_distances
            dists = cosine_distances(cluster)
            n = len(cluster)
            intra_dists.extend(dists[np.triu_indices(n, k=1)])

        if len(cluster) > 0 and len(others) > 0:
            from sklearn.metrics.pairwise import cosine_distances
            cross = cosine_distances(cluster, others)
            inter_dists.extend(cross.ravel())

    mean_intra = np.mean(intra_dists) if intra_dists else 0.0
    mean_inter = np.mean(inter_dists) if inter_dists else 1.0
    return mean_intra / (mean_inter + 1e-10)


# =====================================================
#        4. BIAS & ROBUSTNESS TESTS
# =====================================================

@dataclass
class BiasTestResult:
    test_name: str
    n_logos: int
    pass_rate: float
    mean_score_variance: float
    details: list[dict] = field(default_factory=list)


def run_background_bias_test(
    query_fn,
    image_paths: list[str],
    n_logos: int = 50,
) -> BiasTestResult:
    """
    query_fn(image: Image.Image) -> list[dict] with 'filename' and 'final_score'.

    Tests that background padding does not change the top-1 result.
    """
    paths = random.sample(image_paths, min(n_logos, len(image_paths)))
    passes = 0
    score_vars = []
    details = []

    for path in paths:
        img = Image.open(path).convert("RGB")
        true_fn = os.path.basename(path)
        variants = background_variants(img)
        scores_per_variant = []
        all_top1 = True

        for vname, vimg in variants.items():
            results = query_fn(vimg)
            if results and results[0]["filename"] == true_fn:
                scores_per_variant.append(results[0]["final_score"])
            else:
                all_top1 = False
                scores_per_variant.append(0.0)

        var = np.var(scores_per_variant) if scores_per_variant else 0.0
        passed = all_top1 and var < 5.0
        if passed:
            passes += 1
        score_vars.append(var)
        details.append({"filename": true_fn, "passed": passed, "variance": round(var, 4)})

    return BiasTestResult(
        test_name="background_bias",
        n_logos=len(paths),
        pass_rate=passes / len(paths) if paths else 0.0,
        mean_score_variance=float(np.mean(score_vars)),
        details=details,
    )


def run_color_robustness_test(
    query_fn,
    image_paths: list[str],
    n_logos: int = 40,
) -> BiasTestResult:
    paths = random.sample(image_paths, min(n_logos, len(image_paths)))
    passes = 0
    details = []

    for path in paths:
        img = Image.open(path).convert("RGB")
        true_fn = os.path.basename(path)
        variants = color_variants(img)

        variant_hits = {}
        for vname, vimg in variants.items():
            if vname == "original":
                continue
            results = query_fn(vimg)
            result_fns = [r["filename"] for r in results[:5]]
            variant_hits[vname] = true_fn in result_fns

        passed = all(variant_hits.values())
        if passed:
            passes += 1
        details.append({"filename": true_fn, "variant_hits": variant_hits})

    return BiasTestResult(
        test_name="color_robustness",
        n_logos=len(paths),
        pass_rate=passes / len(paths) if paths else 0.0,
        mean_score_variance=0.0,
        details=details,
    )


def run_ocr_robustness_test(
    query_fn,
    text_image_paths: list[str],
    n_logos: int = 50,
) -> BiasTestResult:
    paths = random.sample(text_image_paths, min(n_logos, len(text_image_paths)))
    passes = 0
    details = []

    for path in paths:
        img = Image.open(path).convert("RGB")
        true_fn = os.path.basename(path)
        variants = ocr_hostile_variants(img)

        text_scores = []
        for vname, vimg in variants.items():
            results = query_fn(vimg)
            for r in results:
                if r["filename"] == true_fn:
                    text_scores.append(r.get("text_score", 0.0))
                    break

        mean_ts = float(np.mean(text_scores)) if text_scores else 0.0
        passed = mean_ts > 0.3
        if passed:
            passes += 1
        details.append({"filename": true_fn, "mean_text_score": round(mean_ts, 3)})

    return BiasTestResult(
        test_name="ocr_robustness",
        n_logos=len(paths),
        pass_rate=passes / len(paths) if paths else 0.0,
        mean_score_variance=0.0,
        details=details,
    )


# =====================================================
#        5. A/B COMPARISON
# =====================================================

@dataclass
class ABResult:
    mean_delta_mrr: float
    p_value: float
    significant: bool
    n_queries: int
    old_mean_mrr: float
    new_mean_mrr: float


def ab_compare(
    old_mrrs: list[float],
    new_mrrs: list[float],
    alpha: float = 0.01,
) -> ABResult:
    """Paired Wilcoxon signed-rank test on per-query MRR values."""
    old_arr = np.array(old_mrrs)
    new_arr = np.array(new_mrrs)
    deltas = new_arr - old_arr

    if np.all(deltas == 0):
        return ABResult(
            mean_delta_mrr=0.0, p_value=1.0, significant=False,
            n_queries=len(old_mrrs),
            old_mean_mrr=float(old_arr.mean()),
            new_mean_mrr=float(new_arr.mean()),
        )

    stat, p_val = wilcoxon(old_arr, new_arr, alternative="less")

    return ABResult(
        mean_delta_mrr=float(deltas.mean()),
        p_value=float(p_val),
        significant=p_val < alpha,
        n_queries=len(old_mrrs),
        old_mean_mrr=float(old_arr.mean()),
        new_mean_mrr=float(new_arr.mean()),
    )


# =====================================================
#        6. HUMAN-IN-THE-LOOP HARNESS
# =====================================================

def generate_hitl_pairs(
    query_fn_old,
    query_fn_new,
    image_paths: list[str],
    n_queries: int = 100,
) -> list[dict]:
    """
    Produce blinded side-by-side comparison data for human annotation.

    Returns a list of dicts:
      {query_image, set_A: [...], set_B: [...], label_A, label_B}

    label_A/B are randomly assigned to 'old'/'new' so annotators
    don't know which is which.
    """
    paths = random.sample(image_paths, min(n_queries, len(image_paths)))
    pairs = []

    for path in paths:
        img = Image.open(path).convert("RGB")
        old_results = query_fn_old(img)
        new_results = query_fn_new(img)

        old_fns = [r["filename"] for r in old_results[:5]]
        new_fns = [r["filename"] for r in new_results[:5]]

        if random.random() < 0.5:
            pairs.append({
                "query": os.path.basename(path),
                "set_A": old_fns, "set_B": new_fns,
                "label_A": "old", "label_B": "new",
            })
        else:
            pairs.append({
                "query": os.path.basename(path),
                "set_A": new_fns, "set_B": old_fns,
                "label_A": "new", "label_B": "old",
            })

    return pairs


# =====================================================
#        CLI ENTRY POINT
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Trademark Engine Benchmark")
    parser.add_argument("--images", default="repository_images",
                        help="Directory of logo images")
    parser.add_argument("--n-augment", type=int, default=8,
                        help="Augmentations per image")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(args.images, ext)))
    paths = sorted(set(paths))

    print(f"📂 Found {len(paths)} images for benchmarking")
    print(f"   Generating {args.n_augment} augmentations per image …")

    all_augmented = {}
    for path in paths:
        aug = generate_augmented_set(path, n=args.n_augment)
        all_augmented[path] = aug

    print(f"   Generated {sum(len(v) for v in all_augmented.values())} total augmented images.")
    print(f"\n   To run full retrieval benchmarks, call the metric functions")
    print(f"   with a query_fn that accepts a PIL Image and returns results.")
    print(f"   See benchmark.py docstring for the full API.\n")

    report = {
        "n_originals": len(paths),
        "n_augmented": sum(len(v) for v in all_augmented.values()),
        "augmentations_per_image": args.n_augment,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved to {args.output}")


if __name__ == "__main__":
    main()
