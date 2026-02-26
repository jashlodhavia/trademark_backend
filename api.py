"""
api.py
──────
FastAPI application for the Trademark Similarity Search API (v3).

Endpoints:
  GET  /                  – health check
  POST /submit-logo       – index a new logo
  POST /similarity-check  – multi-index retrieval + 9-modality re-ranking
  POST /send-email         – send an HTML email via direct MX delivery
"""

import os
import json
import uuid
import shutil
import requests as http_requests
import numpy as np
import imagehash
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymilvus import Collection

from engine import (
    initialize_engines,
    extract_all_features,
    get_visual_embeddings,
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

_milvus_col: Collection | None = None


def _get_collection() -> Collection:
    """Return a cached Milvus collection handle (avoids re-load per request)."""
    global _milvus_col
    if _milvus_col is None:
        _milvus_col = Collection(COLLECTION_NAME)
        _milvus_col.load()
    return _milvus_col


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
    """Generate 2 rotation variants (0° and 180°) for visual robustness."""
    return [img, img.rotate(180, expand=True)]


# ── vectorized cosine similarity helpers ─────────────────────────────────

def _norm_rows(m: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2-D array in-place."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    m /= norms
    return m


def _batch_cosine(query_vecs: np.ndarray, candidate_mat: np.ndarray) -> np.ndarray:
    """
    Vectorized cosine similarity.

    query_vecs:    (Q, D) — e.g. TTA variants
    candidate_mat: (N, D) — all candidates stacked

    Returns (N,) — max similarity across Q query vectors for each candidate.
    """
    q = _norm_rows(query_vecs.astype(np.float64))
    c = _norm_rows(candidate_mat.astype(np.float64))
    sims = q @ c.T  # (Q, N)
    return sims.max(axis=0)


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

    col = _get_collection()
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

        col = _get_collection()

        # ── Stage 1: multi-index candidate generation (parallel) ──
        id_to_entity: dict[int, dict] = {}

        def _collect(hits):
            for h in hits[0]:
                cid = h.id
                if cid not in id_to_entity:
                    id_to_entity[cid] = {
                        f: h.entity.get(f) for f in ALL_OUTPUT_FIELDS
                    }

        def _search_dino():
            return col.search(
                [q_dino.tolist()], "dino_embedding", IP_PARAMS,
                limit=CANDIDATE_LIMIT_VISUAL, output_fields=ALL_OUTPUT_FIELDS,
            )

        def _search_vgg():
            return col.search(
                [q_vgg.tolist()], "vgg_embedding", IP_PARAMS,
                limit=CANDIDATE_LIMIT_VISUAL, output_fields=ALL_OUTPUT_FIELDS,
            )

        def _search_text():
            if not q_has_text:
                return None
            return col.search(
                [q_text_emb.tolist()], "text_embedding", IP_PARAMS,
                limit=CANDIDATE_LIMIT_TEXT, output_fields=ALL_OUTPUT_FIELDS,
            )

        def _search_color():
            return col.search(
                [q_color_hist.tolist()], "color_histogram", IP_PARAMS,
                limit=CANDIDATE_LIMIT_COLOR, output_fields=ALL_OUTPUT_FIELDS,
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            f_dino = pool.submit(_search_dino)
            f_vgg = pool.submit(_search_vgg)
            f_text = pool.submit(_search_text)
            f_color = pool.submit(_search_color)

        _collect(f_dino.result())
        _collect(f_vgg.result())
        text_result = f_text.result()
        if text_result is not None:
            _collect(text_result)
        _collect(f_color.result())

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

        # ── Build candidate matrices for vectorized scoring ──
        c_dino_mat = np.zeros((n, len(q_dino)))
        c_vgg_mat = np.zeros((n, len(q_vgg)))
        c_icon_mat = np.zeros((n, len(q_icon_emb)))
        c_font_mat = np.zeros((n, len(q_font_emb)))
        c_hu_mat = np.zeros((n, len(q_hu)))
        filenames = []
        has_text_flags = []

        for j, cid in enumerate(ids):
            ent = id_to_entity[cid]
            filenames.append(ent["filename"])
            has_text_flags.append(ent["has_text"])
            c_dino_mat[j] = ent["dino_embedding"]
            c_vgg_mat[j] = ent["vgg_embedding"]
            if ent.get("icon_embedding") is not None:
                c_icon_mat[j] = ent["icon_embedding"]
            if ent.get("font_embedding") is not None:
                c_font_mat[j] = ent["font_embedding"]
            if ent.get("hu_moments") is not None:
                c_hu_mat[j] = ent["hu_moments"]

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

        # ── Stage 2: vectorized per-modality scores ──

        # DINO + VGG via TTA (matrix multiply instead of per-candidate loop)
        if tta_dino_vecs is not None and tta_vgg_vecs is not None:
            dino_scores = _batch_cosine(tta_dino_vecs, c_dino_mat)
            vgg_scores = _batch_cosine(tta_vgg_vecs, c_vgg_mat)
        else:
            dino_scores = _batch_cosine(q_dino.reshape(1, -1), c_dino_mat)
            vgg_scores = _batch_cosine(q_vgg.reshape(1, -1), c_vgg_mat)

        # Icon embedding boost (vectorized)
        icon_sims = _batch_cosine(q_icon_emb.reshape(1, -1), c_icon_mat)
        dino_scores = np.maximum(dino_scores, icon_sims)

        # Font similarity (vectorized)
        font_scores = _batch_cosine(q_font_emb.reshape(1, -1), c_font_mat)
        no_font_mask = ~(np.array(has_text_flags, dtype=bool) & q_has_text)
        font_scores[no_font_mask] = 0.0

        # Shape similarity (vectorized)
        shape_scores = _batch_cosine(q_hu.reshape(1, -1), c_hu_mat)

        # Text scoring (skip candidates with no text when query has text)
        text_scores = np.zeros(n)
        for j, cid in enumerate(ids):
            ent = id_to_entity[cid]
            if not q_words and not ent["has_text"]:
                continue
            text_scores[j] = calculate_text_score(q_words, ent["ocr_json"])

        # Color scoring
        color_scores = np.zeros(n)
        for j, cid in enumerate(ids):
            ent = id_to_entity[cid]
            color_scores[j] = color_similarity_emd(
                q_palette_json, ent["color_palette_json"]
            )

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


# ── email (Resend API) ────────────────────────────────────────────────────

SENDER_EMAIL = "onboarding@resend.dev"
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "re_b8SV44CR_Q4Z5eXXMiVFN886m9GRzEq4d")


class EmailRequest(BaseModel):
    receiver_email: str
    subject: str
    body: str


@app.post("/send-email")
async def send_email(req: EmailRequest):
    if "@" not in req.receiver_email:
        raise HTTPException(status_code=400, detail="Invalid email address")

    try:
        resp = http_requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": SENDER_EMAIL,
                "to": [req.receiver_email],
                "subject": req.subject,
                "html": req.body,
            },
            timeout=15,
        )

        if resp.status_code in (200, 201):
            return {
                "status": "sent",
                "receiver": req.receiver_email,
                "subject": req.subject,
            }

        detail = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email sending failed: {str(e)}")
