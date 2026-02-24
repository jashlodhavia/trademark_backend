import os
import re
import json
import warnings
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from difflib import SequenceMatcher
from torchvision import models, transforms
from transformers import AutoProcessor, AutoModel
from paddleocr import PaddleOCR
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# ---------------- SYSTEM OPTIMIZATION ----------------

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

device = "cpu"

# ---------------- CONFIG ----------------

REPO_DIR = "repository_images"
COLLECTION_NAME = "smart_weight_control_v16"
MILVUS_DB_FILE = "./milvus/milvus_smart_v16.db"

TOP_K_MILVUS_RANKING = 50

# ---------------- WEIGHTS ----------------

DINO_FINAL_WEIGHT = 0.60
VGG_FINAL_WEIGHT = 0.25
TEXT_FINAL_WEIGHT = 0.15

assert abs(
    DINO_FINAL_WEIGHT + VGG_FINAL_WEIGHT + TEXT_FINAL_WEIGHT - 1.0
) < 1e-6

VIS_FINAL_WEIGHT = DINO_FINAL_WEIGHT + VGG_FINAL_WEIGHT
DINO_INTERNAL = DINO_FINAL_WEIGHT / VIS_FINAL_WEIGHT
VGG_INTERNAL = VGG_FINAL_WEIGHT / VIS_FINAL_WEIGHT

# ---------------- GLOBAL MODELS ----------------

dino_model = None
dino_processor = None
vgg_model = None
vgg_transform = None
ocr_engine = None


# =====================================================
#                 INITIALIZATION
# =====================================================

def initialize_engines():
    """
    Initialize all ML engines once at startup
    """
    global dino_model, dino_processor, vgg_model, vgg_transform, ocr_engine

    if dino_model is not None:
        # Already initialized
        return

    print("🚀 Initializing Multimodal Trademark Engine...")

    # -------- VGG16 --------
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

    # -------- DINOv2 --------
    dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained(
        "facebook/dinov2-base"
    ).to(device).eval()

    # -------- OCR --------
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False,
    )

    # -------- Milvus --------
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            uri=MILVUS_DB_FILE,
        )

    print("✅ Engines ready.")


# =====================================================
#                     OCR
# =====================================================

def get_ocr_data(image_path):
    """
    Returns:
    - list of words
    - mean confidence
    - OCR json string
    """
    try:
        result = ocr_engine.ocr(image_path, cls=True)

        words, confs, full = [], [], []

        if result and result[0]:
            for line in result[0]:
                txt = re.sub(
                    r"[^a-z0-9]",
                    "",
                    line[1][0].lower(),
                )
                if len(txt) < 3:
                    continue

                conf = float(line[1][1])
                words.append(txt)
                confs.append(conf)
                full.append({
                    "word": txt,
                    "conf": round(conf, 4),
                })

        return (
            words,
            float(np.mean(confs)) if confs else 0.0,
            json.dumps(full),
        )

    except Exception:
        return [], 0.0, "[]"


def calculate_text_score(query_words, ocr_json):
    """
    Fuzzy match OCR words
    """
    try:
        records = json.loads(ocr_json)
        repo_words = [r["word"] for r in records]

        if not query_words or not repo_words:
            return 0.0

        scores = []
        for qw in query_words:
            best = max(
                SequenceMatcher(None, qw, rw).ratio()
                for rw in repo_words
            )
            scores.append(best if best >= 0.6 else 0.0)

        return float(np.mean(scores)) if scores else 0.0

    except Exception:
        return 0.0


# =====================================================
#                  SIMILARITY
# =====================================================

def cosine_sim(a, b):
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    )


# =====================================================
#                  EMBEDDINGS
# =====================================================

@torch.inference_mode()
def get_embeddings(image_paths):
    """
    Returns:
    - fused embedding
    - filenames
    - dino vectors
    - vgg vectors
    """
    images = []
    vgg_images = []
    names = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            vgg_images.append(vgg_transform(img))
            names.append(os.path.basename(path))
        except Exception:
            continue

    if not images:
        return [], [], [], []

    # -------- DINO --------
    dino_inputs = dino_processor(
        images=images,
        return_tensors="pt",
    ).to(device)

    dino_vecs = (
        dino_model(**dino_inputs)
        .last_hidden_state[:, 0, :]
        .cpu()
        .numpy()
    )
    dino_vecs /= np.linalg.norm(
        dino_vecs, axis=1, keepdims=True
    )

    # -------- VGG --------
    vgg_vecs = (
        vgg_model(torch.stack(vgg_images).to(device))
        .cpu()
        .numpy()
    )
    vgg_vecs /= np.linalg.norm(
        vgg_vecs, axis=1, keepdims=True
    )

    # -------- FUSION --------
    fused = np.concatenate(
        [
            dino_vecs * DINO_INTERNAL,
            vgg_vecs * VGG_INTERNAL,
        ],
        axis=1,
    )

    return (
        fused.tolist(),
        names,
        dino_vecs,
        vgg_vecs,
    )


# =====================================================
#               MILVUS COLLECTION
# =====================================================

DINO_DIM = 768
VGG_DIM = 4096
FUSED_DIM = DINO_DIM + VGG_DIM


def ensure_collection():
    """
    Create collection if not exists.
    Stores fused embedding (for search), plus separate DINO & VGG
    vectors and OCR JSON so images are never needed at query time.
    """
    if utility.has_collection(COLLECTION_NAME):
        return

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=FUSED_DIM,
        ),
        FieldSchema(
            name="dino_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=DINO_DIM,
        ),
        FieldSchema(
            name="vgg_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=VGG_DIM,
        ),
        FieldSchema(
            name="filename",
            dtype=DataType.VARCHAR,
            max_length=256,
        ),
        FieldSchema(
            name="ocr_json",
            dtype=DataType.VARCHAR,
            max_length=5000,
        ),
    ]

    schema = CollectionSchema(fields)
    col = Collection(COLLECTION_NAME, schema)

    col.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "L2",
            "index_type": "FLAT",
        },
    )

    print("📦 Milvus collection created (v1 with DINO + VGG fields).")