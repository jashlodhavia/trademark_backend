import os
import uuid
import shutil
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
    REPO_DIR,
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
    Submit a new logo to be stored & indexed
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

    # Embeddings
    vecs, names, _, _ = get_embeddings([save_path])
    if not vecs:
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    # Insert into Milvus
    col = Collection(COLLECTION_NAME)
    col.insert([
        vecs,
        [os.path.basename(save_path)],
        [ocr_json],
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
    Run similarity search and return top 20 matches
    """
    try:
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(status_code=400, detail="Invalid image format")

        temp_path = os.path.join(
            UPLOAD_DIR, f"query_{uuid.uuid4()}_{file.filename}"
        )

        print("Line 113")
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

        print("Line 131")

        hits = col.search(
            q_vec,
            "embedding",
            {"metric_type": "L2"},
            limit=20,
            output_fields=["filename", "ocr_json"],
        )

        print("hits\n\n", len(hits))
        results = []

        for h in hits[0]:
            filename = h.entity.get("filename")
            ocr_json = h.entity.get("ocr_json")

            repo_path = os.path.join(REPO_DIR, filename)
            if not os.path.exists(repo_path):
                continue

            # Compute similarities
            _, _, r_dino, r_vgg = get_embeddings([repo_path])

            dino_s = cosine_sim(q_dino, r_dino[0])
            vgg_s = cosine_sim(q_vgg, r_vgg[0])
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

        print("Line 174")
        
        return {
            "query_ocr": q_words,
            "top_k": len(results),
            "results": results,
        }
    except Exception as e:
        print("Error in similarity check:", e)
        raise HTTPException(status_code=500, detail=str(e))