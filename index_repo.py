"""
index_repo.py
─────────────
Indexes every image in REPO_DIR into Milvus.

For each image the following are stored:
  • fused embedding  (DINO + VGG, used for vector search)
  • dino_embedding   (768‑dim, used for final scoring)
  • vgg_embedding    (4096‑dim, used for final scoring)
  • filename         (basename of the image)
  • ocr_json         (OCR word list with confidences)

After this script finishes, the repository_images folder is
**no longer needed** at query time.
"""

import os
import sys
import glob

from engine import (
    initialize_engines,
    get_embeddings,
    get_ocr_data,
    ensure_collection,
    REPO_DIR,
    COLLECTION_NAME,
)

from pymilvus import Collection


BATCH_SIZE = 50  # images per Milvus insert batch


def index_all_images():
    """Walk REPO_DIR, compute embeddings + OCR, insert into Milvus."""

    # ── 1. Initialize models & Milvus connection ──
    initialize_engines()
    ensure_collection()

    col = Collection(COLLECTION_NAME)

    # ── 2. Gather image paths ──
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(REPO_DIR, ext)))

    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"❌ No images found in '{REPO_DIR}'. Nothing to index.")
        sys.exit(1)

    print(f"📂 Found {len(image_paths)} images in '{REPO_DIR}'")

    # ── 3. Process in batches ──
    total_indexed = 0

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i : i + BATCH_SIZE]

        fused_vecs = []
        dino_vecs = []
        vgg_vecs = []
        filenames = []
        ocr_jsons = []

        for path in batch_paths:
            # Embeddings
            fused, names, dino, vgg = get_embeddings([path])
            if not fused:
                print(f"  ⚠️  Skipping {os.path.basename(path)} (embedding failed)")
                continue

            # OCR
            _, _, ocr_json = get_ocr_data(path)

            fused_vecs.append(fused[0])
            dino_vecs.append(dino[0].tolist())
            vgg_vecs.append(vgg[0].tolist())
            filenames.append(os.path.basename(path))
            ocr_jsons.append(ocr_json)

        if not fused_vecs:
            continue

        # Insert into Milvus (field order must match schema)
        col.insert([
            fused_vecs,       # embedding  (fused)
            dino_vecs,        # dino_embedding
            vgg_vecs,         # vgg_embedding
            filenames,        # filename
            ocr_jsons,        # ocr_json
        ])

        total_indexed += len(fused_vecs)
        print(
            f"  ✅ Indexed batch {i // BATCH_SIZE + 1} "
            f"({total_indexed}/{len(image_paths)} images)"
        )

    # ── 4. Flush & report ──
    col.flush()
    print(f"\n🎉 Done! {total_indexed} images indexed into collection '{COLLECTION_NAME}'.")
    print("   You can now remove the repository_images folder from the deployment.")


if __name__ == "__main__":
    index_all_images()
