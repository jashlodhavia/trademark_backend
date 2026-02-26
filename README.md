# Trademark Similarity Backend

AI-driven trademark infringement detection API: multi-modal similarity search over logos using **DINOv2**, **VGG16**, multilingual **OCR** (PaddleOCR, EasyOCR), **text** (semantic + phonetic), **color**, **font**, **shape**, **icon**, and **pHash**. Built with FastAPI and Milvus.

---

## Table of Contents

- [Running locally](#running-locally)
- [Running with Docker](#running-with-docker)
- [API endpoints](#api-endpoints)
- [Architecture](#architecture)
- [Backend structure](#backend-structure)
- [Deploying to AWS (optional)](#deploying-to-aws-ubuntu-optional)

---

## Running locally

### Prerequisites

- **Python 3.10** (recommended: use `pyenv` or system Python 3.10)
- **pip** and **venv**

### 1. Clone and create virtual environment

```bash
git clone https://github.com/jashlodhavia/trademark_backend.git
cd trademark_backend

python3.10 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

*Note:* First run of `pip install -r requirements.txt` can take several minutes (PyTorch, PaddleOCR, etc.).

### 2. (Optional) Index the repository

Place logo images in `repository_images/` (e.g. PNG/JPEG). Then build the Milvus index:

```bash
python -m index_repo
```

This creates/uses `./milvus/trademark_v18.db` and indexes all images in `repository_images/`. You can run it again later; already-indexed filenames are skipped.

### 3. Start the API server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

- API base: **http://localhost:8000**
- Health: **GET http://localhost:8000/**  
  Example response: `{"status":"ok","service":"trademark-similarity","version":"3.0"}`

### 4. Optional: Resend API key (for `/send-email`)

To use the email endpoint, set the Resend API key:

```bash
export RESEND_API_KEY=re_your_actual_key
```

If unset, `POST /send-email` returns 503.

---

## Running with Docker

### 1. Build the image

```bash
cd trademark_backend
docker build -t trademark_backend:latest .
```

*Tip:* Ensure `repository_images/` and any existing `milvus/` DB are present if you need them inside the container (see volume mount below).

### 2. Run the container

**Minimal run (no persistence, no email):**

```bash
docker run -d --name trademark_backend -p 8000:8000 trademark_backend:latest
```

**With Milvus DB and repository images persisted on the host:**

```bash
mkdir -p ./milvus ./repository_images
docker run -d --name trademark_backend -p 8000:8000 \
  -v "$(pwd)/milvus:/app/milvus" \
  -v "$(pwd)/repository_images:/app/repository_images" \
  trademark_backend:latest
```

**With Resend API key:**

```bash
docker run -d --name trademark_backend -p 8000:8000 \
  -e RESEND_API_KEY=re_your_actual_key \
  -v "$(pwd)/milvus:/app/milvus" \
  -v "$(pwd)/repository_images:/app/repository_images" \
  trademark_backend:latest
```

### 3. Index images (inside container, if not pre-built)

If the DB is empty, run the indexer once:

```bash
docker exec -it trademark_backend python -m index_repo
```

### 4. Verify

```bash
curl http://localhost:8000/
# Expect: {"status":"ok","service":"trademark-similarity","version":"3.0"}
```

### 5. Stop / remove

```bash
docker stop trademark_backend
docker rm trademark_backend
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check. |
| POST | `/similarity-check` | Upload a logo image; returns top-K similar trademarks with per-modality scores (DINO, VGG, text, color, font, shape). |
| POST | `/submit-logo` | Upload a logo image; indexes it into the Milvus repository. |
| POST | `/send-email` | Send an HTML email (body: `receiver_email`, `subject`, `body`). Requires `RESEND_API_KEY`. |

Example (similarity check):

```bash
curl -X POST http://localhost:8000/similarity-check -F "file=@/path/to/logo.png"
```

---

## Architecture

High-level flow:

1. **Query:** User uploads a logo image to the API.
2. **Preprocess:** Background removal (rembg / U2-Net), pad and resize.
3. **Feature extraction:** OCR (Paddle + Easy), DINOv2 + VGG16 embeddings, color histogram/palette, shape (Hu moments), icon embedding, font embedding (DINO on text crops), pHash.
4. **Retrieval:** Parallel Milvus searches on DINO, VGG, text, and color indices; candidate IDs are merged.
5. **Scoring:** Per-candidate scores for text, color, font, shape; TTA (0°, 180°) for DINO/VGG; icon and pHash boost.
6. **Fusion:** Adaptive weights, z-score normalization, Reciprocal Rank Fusion; perfect-text and soft-text overrides; top-K results returned.

```
┌─────────────┐     ┌──────────────────────────────────────────────────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI (api.py)                                         │────▶│   Milvus    │
│  (e.g. UI)  │     │  • /similarity-check  • /submit-logo  • /send-email        │     │  Vector DB  │
└─────────────┘     │                                                             │     │ trademark_  │
                    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │     │   v18       │
                    │  │ engine.py   │  │ text_engine │  │ color_engine       │  │     └─────────────┘
                    │  │ DINO, VGG   │  │ OCR, embed  │  │ shape_engine        │  │
                    │  │ icon, phash │  │ text score  │  │ fusion.py           │  │
                    │  │ preprocessing│  │ font embed │  │ rerank, weights     │  │
                    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
                    └──────────────────────────────────────────────────────────┘
```

---

## Backend structure

```
trademark_backend/
├── api.py              # FastAPI app: routes, TTA, vectorized scoring, Resend email
├── engine.py           # DINO/VGG init, visual embeddings, Milvus schema, extract_all_features
├── preprocessing.py    # rembg (U2-Net), pad_and_resize, extract_foreground
├── text_engine.py      # PaddleOCR, EasyOCR, sentence-transformers, text score, font embedding
├── color_engine.py     # Color palette (K-means), histogram, EMD similarity
├── shape_engine.py     # Hu moments, contour histogram, shape_similarity
├── fusion.py           # Z-score, RRF, adaptive weights, rerank, CandidateScores
├── index_repo.py       # Batch indexing: repository_images → Milvus (run as __main__)
├── benchmark.py       # Evaluation helpers (Hit@K, MRR, augmentation, bias tests)
├── requirements.txt
├── Dockerfile
├── repository_images/ # Input images to index (optional at runtime if DB pre-built)
├── milvus/             # Milvus DB file (trademark_v18.db) created by index_repo
├── uploads/            # Temp uploads (created at runtime)
└── docs/
    ├── BRD_Trademark_Infringement_Detection.md
    └── Project_Report_Trademark_Infringement_Detection.md
```

| Module | Role |
|--------|------|
| **api.py** | HTTP API; multi-index search (parallel); batch cosine scoring; calls engine, text_engine, color_engine, shape_engine, fusion. |
| **engine.py** | Loads DINOv2, VGG16; `extract_all_features()` (preprocess → OCR → visual/icon/color/shape/phash); Milvus connection and schema. |
| **preprocessing.py** | Background removal (rembg), pad-and-resize for model input. |
| **text_engine.py** | Multilingual OCR, transliteration/translation, text and font embeddings, hybrid text similarity (edit + semantic + phonetic). |
| **color_engine.py** | CIELAB palette, histogram, EMD-based color similarity. |
| **shape_engine.py** | Hu moments and contour angle histogram from edges; shape_similarity. |
| **fusion.py** | Normalization, adaptive weights, RRF, overrides; produces final ranking. |
| **index_repo.py** | Scans `repository_images/`, calls `extract_all_features`, inserts into Milvus in batches; skips already-indexed filenames. |

---

## Deploying to AWS Ubuntu (optional)

These steps assume an EC2 instance (e.g. Ubuntu 22.04) with SSH access.

### 1. Connect and install Docker

```bash
ssh -i /path/to/your-key.pem ubuntu@<EC2_PUBLIC_IP>

sudo apt update
sudo apt install -y docker.io git
sudo systemctl enable --now docker
```

### 2. Clone and build

```bash
mkdir -p ~/trademark_backend && cd ~/trademark_backend
git clone https://github.com/jashlodhavia/trademark_backend.git .

sudo docker build -t trademark_backend:latest .
```

### 3. Run with volume and env

```bash
sudo docker run -d \
  --name trademark_backend \
  -p 8000:8000 \
  -e RESEND_API_KEY=re_your_actual_key \
  -v "$(pwd)/milvus:/app/milvus" \
  -v "$(pwd)/repository_images:/app/repository_images" \
  trademark_backend:latest
```

### 4. Index (if needed)

```bash
sudo docker exec -it trademark_backend python -m index_repo
```

### 5. Verify

```bash
curl http://localhost:8000/
# From your machine: curl http://<EC2_PUBLIC_IP>:8000/
```

For production, consider a process manager (e.g. systemd wrapping `docker run`), a reverse proxy (Nginx), and HTTPS.

---

*For full project context, BRD, and report (architecture, 5-year roadmap, AI/ML components), see the `docs/` folder.*
