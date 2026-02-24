import os
import uuid
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pymilvus import Collection

# import from engine.py
from engine import (
    initialize_engines,
    get_embeddings,
    get_ocr_data,
    calculate_text_score,
    cosine_sim,
    DINO_FINAL_WEIGHT,
    VGG_FINAL_WEIGHT,
    TEXT_FINAL_WEIGHT,
    COLLECTION_NAME,
)

# ---------------- APP CONFIG ----------------

app = FastAPI(
    title="Trademark Similarity Search API",
    version="1.0.0",
)

# Allow frontend (Vercel etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- STARTUP ----------------

@app.on_event("startup")
def startup_event():
    """
    Initialize all heavy engines once when container starts
    """
    initialize_engines()

# ---------------- HEALTH CHECK ----------------

@app.get("/")
def health_check():
    return {"status": "ok", "service": "trademark-similarity"}

# ---------------- API 1: SUBMIT LOGO ----------------

@app.post("/submit-logo")
async def submit_logo(file: UploadFile = File(...)):
    """
    Submit a new logo to be stored & indexed.
    Stores fused, DINO, and VGG embeddings + OCR in Milvus.
    """
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid image format")

    uid = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{uid}_{file.filename}")

    # Save file
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # OCR
    words, conf, ocr_json = get_ocr_data(save_path)

    # Embeddings (fused + individual DINO & VGG)
    fused, names, dino_vecs, vgg_vecs = get_embeddings([save_path])
    if not fused:
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    # Insert into Milvus (field order must match schema)
    col = Collection(COLLECTION_NAME)
    col.insert([
        fused,                              # embedding  (fused)
        [dino_vecs[0].tolist()],            # dino_embedding
        [vgg_vecs[0].tolist()],             # vgg_embedding
        [os.path.basename(save_path)],      # filename
        [ocr_json],                         # ocr_json
    ])
    col.flush()

    return {
        "submission_id": uid,
        "filename": file.filename,
        "ocr_words": words,
        "ocr_confidence": round(conf, 3),
        "status": "submitted",
    }

# ---------------- API 2: SIMILARITY CHECK ----------------

@app.post("/similarity-check")
async def similarity_check(file: UploadFile = File(...)):
    """
    Run similarity search and return top 20 matches.
    All scoring data is read from Milvus — no image folder needed.
    """
    try:
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(status_code=400, detail="Invalid image format")

        temp_path = os.path.join(
            UPLOAD_DIR, f"query_{uuid.uuid4()}_{file.filename}"
        )

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # OCR query
        q_words, _, _ = get_ocr_data(temp_path)

        # Query embeddings
        q_vec, _, q_dino, q_vgg = get_embeddings([temp_path])
        if not q_vec:
            raise HTTPException(status_code=500, detail="Query embedding failed")

        q_dino = q_dino[0]
        q_vgg = q_vgg[0]

        col = Collection(COLLECTION_NAME)
        col.load()

        # Search by fused vector and retrieve stored DINO/VGG vectors + OCR
        hits = col.search(
            q_vec,
            "embedding",
            {"metric_type": "L2"},
            limit=20,
            output_fields=[
                "filename",
                "ocr_json",
                "dino_embedding",
                "vgg_embedding",
            ],
        )

        results = []

        for h in hits[0]:
            filename = h.entity.get("filename")
            ocr_json = h.entity.get("ocr_json")
            r_dino = np.array(h.entity.get("dino_embedding"))
            r_vgg = np.array(h.entity.get("vgg_embedding"))

            dino_s = cosine_sim(q_dino, r_dino)
            vgg_s = cosine_sim(q_vgg, r_vgg)
            text_s = calculate_text_score(q_words, ocr_json)

            final_score = (
                dino_s * DINO_FINAL_WEIGHT
                + vgg_s * VGG_FINAL_WEIGHT
                + text_s * TEXT_FINAL_WEIGHT
            ) * 100

            results.append({
                "filename": filename,
                "final_score": round(final_score, 2),
                "dino_score": round(dino_s, 3),
                "vgg_score": round(vgg_s, 3),
                "text_score": round(text_s, 3),
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)

        # Clean up temp query file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "query_ocr": q_words,
            "top_k": len(results),
            "results": results,
        }
    except Exception as e:
        print("Error in similarity check:", e)
        raise HTTPException(status_code=500, detail=str(e))