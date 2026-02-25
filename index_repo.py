"""
index_repo.py
─────────────
Indexes every image in REPO_DIR into Milvus (v3 schema).

For each image the following are extracted and stored:
  • dino_embedding     (768-dim,  attention-weighted, on foreground)
  • vgg_embedding      (4096-dim, on foreground)
  • text_embedding     (384-dim,  multilingual sentence-transformer)
  • font_embedding     (768-dim,  DINOv2 on cropped text regions)
  • color_histogram    (48-dim,   CIELAB histogram)
  • shape_histogram    (64-dim,   contour angle histogram)
  • hu_moments         (7-dim,    log-scale Hu Moments)
  • icon_embedding     (768-dim,  DINO on icon-only region)
  • color_palette_json (JSON,     dominant colors for EMD re-ranking)
  • ocr_json           (JSON,     per-word OCR with confidence)
  • ocr_text_raw       (string,   concatenated raw OCR text)
  • phash_hex          (string,   perceptual hash hex)
  • has_text           (bool)
  • filename

After indexing, the repository_images folder is **not needed** at query time.
"""

import os
import sys
import glob

from engine import (
    initialize_engines,
    ensure_collection,
    extract_all_features,
    REPO_DIR,
    COLLECTION_NAME,
)

from pymilvus import Collection


BATCH_SIZE = 50


def _get_existing_filenames(col: Collection) -> set[str]:
    """Query Milvus for all filenames already indexed."""
    try:
        col.load()
        results = col.query(expr="id >= 0", output_fields=["filename"])
        return {r["filename"] for r in results}
    except Exception:
        return set()


def index_all_images():
    initialize_engines()
    ensure_collection()

    col = Collection(COLLECTION_NAME)

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(REPO_DIR, ext)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"❌ No images found in '{REPO_DIR}'. Nothing to index.")
        sys.exit(1)

    existing = _get_existing_filenames(col)
    if existing:
        before = len(image_paths)
        image_paths = [p for p in image_paths if os.path.basename(p) not in existing]
        print(f"📂 Found {before} images, {len(existing)} already indexed, {len(image_paths)} remaining")
    else:
        print(f"📂 Found {len(image_paths)} images in '{REPO_DIR}'")

    if not image_paths:
        print("✅ All images already indexed. Nothing to do.")
        return

    total_indexed = 0

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i : i + BATCH_SIZE]

        rows = {
            "dino_embedding": [],
            "vgg_embedding": [],
            "text_embedding": [],
            "font_embedding": [],
            "color_histogram": [],
            "shape_histogram": [],
            "hu_moments": [],
            "icon_embedding": [],
            "color_palette_json": [],
            "filename": [],
            "ocr_json": [],
            "ocr_text_raw": [],
            "phash_hex": [],
            "has_text": [],
        }

        for path in batch_paths:
            feats = extract_all_features(path)
            if feats is None:
                print(f"  ⚠️  Skipping {os.path.basename(path)}")
                continue

            feats.pop("_fg_img", None)

            for key in rows:
                rows[key].append(feats[key])

        if not rows["filename"]:
            continue

        col.insert([
            rows["dino_embedding"],
            rows["vgg_embedding"],
            rows["text_embedding"],
            rows["font_embedding"],
            rows["color_histogram"],
            rows["shape_histogram"],
            rows["hu_moments"],
            rows["icon_embedding"],
            rows["color_palette_json"],
            rows["filename"],
            rows["ocr_json"],
            rows["ocr_text_raw"],
            rows["phash_hex"],
            rows["has_text"],
        ])

        total_indexed += len(rows["filename"])
        print(
            f"  ✅ Indexed batch {i // BATCH_SIZE + 1} "
            f"({total_indexed}/{len(image_paths)} images)"
        )

    col.flush()
    print(f"\n🎉 Done! {total_indexed} images indexed into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    index_all_images()
