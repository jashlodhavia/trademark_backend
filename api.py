"""
api.py
──────
FastAPI application for the Trademark Similarity Search API (v3).

Endpoints:
  GET  /                  – health check
  POST /submit-logo       – index a new logo
  POST /similarity-check  – multi-index retrieval + 9-modality re-ranking
"""

import os
import json
import uuid
import shutil
import numpy as np
import imagehash
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymilvus import Collection

from engine import (
    initialize_engines,
    extract_all_features,
    get_visual_embeddings,
    cosine_sim,
    COLLECTION_NAME,
)
from preprocessing import preprocess_image
from text_engine import (
    calculate_text_score,
    batch_precompute_embeddings,
    clear_word_cache,
)
from color_engine import color_similarity_emd
from shape_engine import shape_similarity
from fusion import CandidateScores, rerank


# ── app config ────────────────────────────────────────────────────────────

app = FastAPI(title="Trademark Similarity Search API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CANDIDATE_LIMIT_VISUAL = 50
CANDIDATE_LIMIT_TEXT = 50
CANDIDATE_LIMIT_COLOR = 30
TOP_K_RESULTS = 20

IP_PARAMS = {"metric_type": "IP"}

ALL_OUTPUT_FIELDS = [
    "filename", "dino_embedding", "vgg_embedding",
    "text_embedding", "font_embedding",
    "color_histogram", "color_palette_json",
    "ocr_json", "ocr_text_raw", "has_text",
    "shape_histogram", "hu_moments",
    "icon_embedding", "phash_hex",
]


# ── startup ───────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    initialize_engines()


# ── health ────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "service": "trademark-similarity", "version": "3.0"}


# ── TTA (test-time augmentation) ─────────────────────────────────────────

def _tta_variants(img: Image.Image) -> list[Image.Image]:
    """Generate 4 rotation variants for visual robustness."""
    return [img.rotate(angle, expand=True) for angle in [0, 90, 180, 270]]


# ── submit logo ───────────────────────────────────────────────────────────

@app.post("/submit-logo")
async def submit_logo(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid image format")

    uid = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{uid}_{file.filename}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    feats = extract_all_features(save_path)
    if feats is None:
        raise HTTPException(status_code=500, detail="Feature extraction failed")

    feats.pop("_fg_img", None)

    col = Collection(COLLECTION_NAME)
    col.insert([[feats[k]] for k in [
        "dino_embedding", "vgg_embedding", "text_embedding",
        "font_embedding", "color_histogram",
        "shape_histogram", "hu_moments", "icon_embedding",
        "color_palette_json", "filename", "ocr_json",
        "ocr_text_raw", "phash_hex", "has_text",
    ]])
    col.flush()

    return {
        "submission_id": uid,
        "filename": file.filename,
        "ocr_words": feats["ocr_text_raw"].split() if feats["ocr_text_raw"] else [],
        "has_text": feats["has_text"],
        "status": "submitted",
    }


# ── similarity check ─────────────────────────────────────────────────────

@app.post("/similarity-check")
async def similarity_check(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(status_code=400, detail="Invalid image format")

        temp_path = os.path.join(
            UPLOAD_DIR, f"query_{uuid.uuid4()}_{file.filename}"
        )
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ── extract query features ──
        q_feats = extract_all_features(temp_path)
        if q_feats is None:
            raise HTTPException(
                status_code=500, detail="Query feature extraction failed"
            )

        fg_img = q_feats.pop("_fg_img", None)

        q_dino = np.array(q_feats["dino_embedding"])
        q_vgg = np.array(q_feats["vgg_embedding"])
        q_text_emb = np.array(q_feats["text_embedding"])
        q_color_hist = np.array(q_feats["color_histogram"])
        q_font_emb = np.array(q_feats["font_embedding"])
        q_hu = np.array(q_feats["hu_moments"])
        q_icon_emb = np.array(q_feats["icon_embedding"])
        q_has_text = q_feats["has_text"]
        q_words = q_feats["ocr_text_raw"].split() if q_feats["ocr_text_raw"] else []
        q_palette_json = q_feats["color_palette_json"]
        q_phash_hex = q_feats["phash_hex"]

        # ── TTA variants for visual scoring ──
        tta_dino_vecs = None
        tta_vgg_vecs = None
        if fg_img is not None:
            tta_imgs = _tta_variants(fg_img)
            tta_dino_vecs, tta_vgg_vecs = get_visual_embeddings(tta_imgs)

        col = Collection(COLLECTION_NAME)
        col.load()

        # ── Stage 1: multi-index candidate generation ──
        candidate_ids: set[int] = set()
        id_to_entity: dict[int, dict] = {}

        def _collect(hits):
            for h in hits[0]:
                cid = h.id
                if cid not in id_to_entity:
                    candidate_ids.add(cid)
                    id_to_entity[cid] = {
                        f: h.entity.get(f) for f in ALL_OUTPUT_FIELDS
                    }

        dino_hits = col.search(
            [q_dino.tolist()], "dino_embedding", IP_PARAMS,
            limit=CANDIDATE_LIMIT_VISUAL, output_fields=ALL_OUTPUT_FIELDS,
        )
        _collect(dino_hits)

        vgg_hits = col.search(
            [q_vgg.tolist()], "vgg_embedding", IP_PARAMS,
            limit=CANDIDATE_LIMIT_VISUAL, output_fields=ALL_OUTPUT_FIELDS,
        )
        _collect(vgg_hits)

        if q_has_text:
            text_hits = col.search(
                [q_text_emb.tolist()], "text_embedding", IP_PARAMS,
                limit=CANDIDATE_LIMIT_TEXT, output_fields=ALL_OUTPUT_FIELDS,
            )
            _collect(text_hits)

        color_hits = col.search(
            [q_color_hist.tolist()], "color_histogram", IP_PARAMS,
            limit=CANDIDATE_LIMIT_COLOR, output_fields=ALL_OUTPUT_FIELDS,
        )
        _collect(color_hits)

        # ── Batch precompute word embeddings ──
        clear_word_cache()
        ids = list(id_to_entity.keys())
        n = len(ids)

        all_words = list(q_words)
        for cid in ids:
            ent = id_to_entity[cid]
            try:
                records = json.loads(ent["ocr_json"])
                for r in records:
                    all_words.extend([
                        r.get("word", ""),
                        r.get("transliterated", ""),
                        r.get("translated", ""),
                    ])
            except Exception:
                pass
        batch_precompute_embeddings(all_words)

        # ── pHash pre-filter ──
        phash_boost = np.zeros(n)
        try:
            q_phash = imagehash.hex_to_hash(q_phash_hex)
            for j, cid in enumerate(ids):
                r_phash_hex = id_to_entity[cid].get("phash_hex", "")
                if r_phash_hex:
                    r_phash = imagehash.hex_to_hash(r_phash_hex)
                    hamming = q_phash - r_phash
                    if hamming <= 8:
                        phash_boost[j] = 1.0 - (hamming / 256.0)
        except Exception:
            pass

        # ── Stage 2: compute per-modality scores ──
        dino_scores = np.zeros(n)
        vgg_scores = np.zeros(n)
        text_scores = np.zeros(n)
        color_scores = np.zeros(n)
        font_scores = np.zeros(n)
        shape_scores = np.zeros(n)
        filenames = []
        has_text_flags = []

        for j, cid in enumerate(ids):
            ent = id_to_entity[cid]
            filenames.append(ent["filename"])
            c_has_text = ent["has_text"]
            has_text_flags.append(c_has_text)

            r_dino = np.array(ent["dino_embedding"])
            r_vgg = np.array(ent["vgg_embedding"])

            # TTA: max similarity across 8 rotated/flipped variants
            if tta_dino_vecs is not None and tta_vgg_vecs is not None:
                dino_scores[j] = max(
                    cosine_sim(v, r_dino) for v in tta_dino_vecs
                )
                vgg_scores[j] = max(
                    cosine_sim(v, r_vgg) for v in tta_vgg_vecs
                )
            else:
                dino_scores[j] = cosine_sim(q_dino, r_dino)
                vgg_scores[j] = cosine_sim(q_vgg, r_vgg)

            # Icon embedding boost
            r_icon = ent.get("icon_embedding")
            if r_icon is not None:
                icon_sim = cosine_sim(q_icon_emb, np.array(r_icon))
                dino_scores[j] = max(dino_scores[j], icon_sim)

            text_scores[j] = calculate_text_score(
                q_words, ent["ocr_json"]
            )

            color_scores[j] = color_similarity_emd(
                q_palette_json, ent["color_palette_json"]
            )

            if q_has_text and c_has_text:
                r_font = np.array(ent["font_embedding"])
                font_scores[j] = cosine_sim(q_font_emb, r_font)

            # Shape similarity
            r_hu = ent.get("hu_moments")
            if r_hu is not None:
                shape_scores[j] = shape_similarity(q_hu, np.array(r_hu))

        # ── Stage 3: fusion + re-ranking ──
        candidates = CandidateScores(
            ids=ids,
            filenames=filenames,
            dino=dino_scores,
            vgg=vgg_scores,
            text=text_scores,
            color=color_scores,
            font=font_scores,
            shape=shape_scores,
            has_text=has_text_flags,
            phash_boost=phash_boost,
        )

        results = rerank(candidates, q_has_text, top_k=TOP_K_RESULTS)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "query_ocr": q_words,
            "query_has_text": q_has_text,
            "candidates_evaluated": n,
            "top_k": len(results),
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Error in similarity check:", e)
        raise HTTPException(status_code=500, detail=str(e))
