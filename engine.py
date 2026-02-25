"""
engine.py
─────────
Core model initialization (DINO, VGG), visual embedding extraction
with preprocessing, attention-weighted DINOv2, Milvus schema (v3
with 9 modalities), pHash, icon/text splitting, and cosine similarity.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn

import cv2
import imagehash
from PIL import Image
from torchvision import models, transforms
from transformers import AutoProcessor, AutoModel
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from preprocessing import preprocess_image
from text_engine import (
    initialize_text_engines,
    get_ocr_data,
    get_text_embedding,
    extract_english_words,
    crop_text_regions,
    get_font_embedding,
    TEXT_EMBED_DIM,
    FONT_EMBED_DIM,
)
from color_engine import (
    extract_color_palette,
    palette_to_json,
    compute_color_histogram,
    COLOR_HIST_DIM,
)
from shape_engine import (
    extract_shape_features,
    SHAPE_HIST_DIM,
    HU_DIM,
)

# ── system optimization ───────────────────────────────────────────────────

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

device = "cpu"

# ── config ────────────────────────────────────────────────────────────────

REPO_DIR = "repository_images"
COLLECTION_NAME = "trademark_v18"
MILVUS_DB_FILE = "./milvus/trademark_v18.db"

# ── dimensions ────────────────────────────────────────────────────────────

DINO_DIM = 768
VGG_DIM = 4096

# ── global models ─────────────────────────────────────────────────────────

dino_model = None
dino_processor = None
vgg_model = None
vgg_transform = None


# =====================================================
#                 INITIALIZATION
# =====================================================

def initialize_engines():
    global dino_model, dino_processor, vgg_model, vgg_transform

    if dino_model is not None:
        return

    print("🚀 Initializing Multimodal Trademark Engine v3 …")

    # ── VGG16 ──
    vgg_model = models.vgg16(weights="DEFAULT").to(device)
    vgg_model.classifier = nn.Sequential(
        *list(vgg_model.classifier.children())[:-1]
    )
    vgg_model.eval()

    vgg_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ── DINOv2 ──
    dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained(
        "facebook/dinov2-base"
    ).to(device).eval()

    # ── Text / OCR / Sentence-Transformer ──
    initialize_text_engines()

    # ── Milvus ──
    if not connections.has_connection("default"):
        connections.connect(alias="default", uri=MILVUS_DB_FILE)

    print("✅ All engines ready.")


# =====================================================
#            COSINE SIMILARITY
# =====================================================

def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    )


# =====================================================
#        VISUAL EMBEDDINGS (with preprocessing)
# =====================================================

@torch.inference_mode()
def get_visual_embeddings(images: list[Image.Image]):
    """
    Compute DINO (attention-weighted) and VGG embeddings for a list
    of **already-preprocessed** PIL images.

    Returns:
        dino_vecs – (N, 768) np.ndarray, L2-normalized
        vgg_vecs  – (N, 4096) np.ndarray, L2-normalized
    """
    if not images:
        return np.empty((0, DINO_DIM)), np.empty((0, VGG_DIM))

    # ── DINO (attention-weighted CLS with fallback) ──
    dino_inputs = dino_processor(
        images=images, return_tensors="pt"
    ).to(device)

    outputs = dino_model(**dino_inputs, output_attentions=True)

    if outputs.attentions is not None and len(outputs.attentions) > 0:
        last_attn = outputs.attentions[-1]
        cls_attn = last_attn[:, :, 0, 1:].mean(dim=1)
        cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-10)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        dino_vecs = (patch_tokens * cls_attn.unsqueeze(-1)).sum(dim=1)
    else:
        dino_vecs = outputs.last_hidden_state[:, 0, :]

    dino_vecs = dino_vecs.cpu().numpy()

    norms = np.linalg.norm(dino_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    dino_vecs /= norms

    # ── VGG ──
    vgg_batch = torch.stack([vgg_transform(img) for img in images]).to(device)
    vgg_vecs = vgg_model(vgg_batch).cpu().numpy()

    norms = np.linalg.norm(vgg_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    vgg_vecs /= norms

    return dino_vecs, vgg_vecs


# =====================================================
#          PERCEPTUAL HASH (pHash)
# =====================================================

def compute_phash(image: Image.Image) -> str:
    """Compute perceptual hash (256-bit) and return as hex string."""
    return str(imagehash.phash(image, hash_size=16))


# =====================================================
#          ICON / TEXT REGION SPLITTING
# =====================================================

def split_icon_text(image: Image.Image, raw_ocr_result) -> Image.Image:
    """
    Mask out text bounding boxes (to gray) to isolate the icon region.
    Returns the icon-only image for separate DINO embedding.
    """
    img_array = np.array(image)
    text_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)

    if raw_ocr_result and raw_ocr_result[0]:
        for line in raw_ocr_result[0]:
            bbox = np.array(line[0], dtype=np.int32)
            cv2.fillPoly(text_mask, [bbox], 255)

    icon = img_array.copy()
    icon[text_mask > 0] = [128, 128, 128]
    return Image.fromarray(icon)


# =====================================================
#        FULL FEATURE EXTRACTION (single image)
# =====================================================

def extract_all_features(image_path: str) -> dict | None:
    """
    End-to-end feature extraction for one image:
      preprocessing → visual embeddings → OCR → text embedding →
      font embedding → color histogram + palette → shape → icon → phash

    Returns a dict with all fields needed for Milvus insertion, or
    None if the image cannot be processed.
    """
    try:
        raw_img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    # ── preprocessing ──
    fg_img = preprocess_image(raw_img)

    # ── visual embeddings ──
    dino_vecs, vgg_vecs = get_visual_embeddings([fg_img])
    if dino_vecs.shape[0] == 0:
        return None

    dino_vec = dino_vecs[0]
    vgg_vec = vgg_vecs[0]

    # ── OCR (on raw image, not preprocessed) ──
    words, mean_conf, ocr_json, raw_ocr = get_ocr_data(image_path)
    has_text = len(words) > 0

    # ── Truncate OCR JSON if it exceeds Milvus VARCHAR limit ──
    OCR_JSON_MAX = 15500
    if len(ocr_json) > OCR_JSON_MAX:
        import json as _json
        try:
            records = _json.loads(ocr_json)
            while len(_json.dumps(records, ensure_ascii=False)) > OCR_JSON_MAX and records:
                records.pop()
            ocr_json = _json.dumps(records, ensure_ascii=False)
        except Exception:
            ocr_json = ocr_json[:OCR_JSON_MAX]

    # ── English representation for embedding + ocr_text_raw ──
    english_words = extract_english_words(ocr_json)
    text_emb = get_text_embedding(english_words)
    ocr_text_raw = " ".join(english_words) if english_words else ""

    # ── font embedding ──
    text_regions = crop_text_regions(raw_img, raw_ocr)
    font_emb = get_font_embedding(text_regions, dino_processor, dino_model, device)

    # ── color ──
    centers, proportions = extract_color_palette(fg_img)
    color_hist = compute_color_histogram(fg_img)
    palette_json = palette_to_json(centers, proportions)

    # ── shape ──
    hu_moments, shape_hist = extract_shape_features(fg_img)

    # ── icon embedding (text regions masked out) ──
    icon_img = split_icon_text(raw_img, raw_ocr)
    icon_preprocessed = preprocess_image(icon_img)
    icon_dino_vecs, _ = get_visual_embeddings([icon_preprocessed])
    icon_emb = icon_dino_vecs[0] if icon_dino_vecs.shape[0] > 0 else np.zeros(DINO_DIM, dtype=np.float32)

    # ── perceptual hash ──
    phash_hex = compute_phash(raw_img)

    return {
        "filename": os.path.basename(image_path),
        "dino_embedding": dino_vec.tolist(),
        "vgg_embedding": vgg_vec.tolist(),
        "text_embedding": text_emb.tolist(),
        "font_embedding": font_emb.tolist(),
        "color_histogram": color_hist.tolist(),
        "color_palette_json": palette_json,
        "ocr_json": ocr_json,
        "ocr_text_raw": ocr_text_raw,
        "has_text": has_text,
        "shape_histogram": shape_hist.tolist(),
        "hu_moments": hu_moments.tolist(),
        "icon_embedding": icon_emb.tolist(),
        "phash_hex": phash_hex,
        "_fg_img": fg_img,
    }


# =====================================================
#               MILVUS COLLECTION (v3)
# =====================================================

def ensure_collection():
    """
    Create the v3 collection if it does not exist.

    Stores nine embedding vectors (DINO, VGG, text, font, color,
    shape, hu_moments, icon) plus pHash and metadata.
    """
    if utility.has_collection(COLLECTION_NAME):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=True),
        FieldSchema(name="dino_embedding", dtype=DataType.FLOAT_VECTOR,
                    dim=DINO_DIM),
        FieldSchema(name="vgg_embedding", dtype=DataType.FLOAT_VECTOR,
                    dim=VGG_DIM),
        FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR,
                    dim=TEXT_EMBED_DIM),
        FieldSchema(name="font_embedding", dtype=DataType.FLOAT_VECTOR,
                    dim=FONT_EMBED_DIM),
        FieldSchema(name="color_histogram", dtype=DataType.FLOAT_VECTOR,
                    dim=COLOR_HIST_DIM),
        FieldSchema(name="shape_histogram", dtype=DataType.FLOAT_VECTOR,
                    dim=SHAPE_HIST_DIM),
        FieldSchema(name="hu_moments", dtype=DataType.FLOAT_VECTOR,
                    dim=HU_DIM),
        FieldSchema(name="icon_embedding", dtype=DataType.FLOAT_VECTOR,
                    dim=DINO_DIM),
        FieldSchema(name="color_palette_json", dtype=DataType.VARCHAR,
                    max_length=1000),
        FieldSchema(name="filename", dtype=DataType.VARCHAR,
                    max_length=256),
        FieldSchema(name="ocr_json", dtype=DataType.VARCHAR,
                    max_length=32000),
        FieldSchema(name="ocr_text_raw", dtype=DataType.VARCHAR,
                    max_length=2000),
        FieldSchema(name="phash_hex", dtype=DataType.VARCHAR,
                    max_length=128),
        FieldSchema(name="has_text", dtype=DataType.BOOL),
    ]

    schema = CollectionSchema(fields)
    col = Collection(COLLECTION_NAME, schema)

    ip_flat = {"metric_type": "IP", "index_type": "FLAT"}
    col.create_index(field_name="dino_embedding", index_params=ip_flat)
    col.create_index(field_name="vgg_embedding", index_params=ip_flat)
    col.create_index(field_name="text_embedding", index_params=ip_flat)
    col.create_index(field_name="font_embedding", index_params=ip_flat)
    col.create_index(field_name="color_histogram", index_params=ip_flat)
    col.create_index(field_name="shape_histogram", index_params=ip_flat)
    col.create_index(field_name="icon_embedding", index_params=ip_flat)

    print("📦 Milvus collection created (v3 — 9-modality schema).")
