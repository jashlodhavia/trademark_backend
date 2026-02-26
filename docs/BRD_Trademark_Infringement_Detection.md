# Business Requirements Document (BRD)
## AI-Driven Trademark Infringement Detection Framework

**Document version:** 1.0  
**Last updated:** [Date]

---

## 1. Executive Summary

### Problem
Government intellectual property (IP) management bodies are facing a sharp rise in trademark infringement litigations. This trend is driven by the widespread use of AI-powered tools that enable easy image generation, content creation, and brand mimicry. Detecting infringement risks manually is no longer scalable or consistent.

### Objective
To deliver a **scalable, AI-enabled solution** that proactively detects potential trademark infringements across digital content, helping a government IP management department reduce litigation risk, enhance compliance, and protect its intellectual property portfolio. The solution applies **machine learning (ML) and artificial intelligence (AI) throughout**—in computer vision (logo and image similarity), natural language processing (multilingual text and brand matching), and intelligent fusion of multiple modalities—as appropriate for an AI Applied Project.

### Scope and Deliverables
- **In scope:** Design and implementation of a detection framework that supports (1) identical or similar names in different languages, (2) replication of trademarked images with modified orientation or aspect, (3) synonyms or phonetically similar brand names, and (4) visual or verbal mimicry that may cause consumer confusion.
- **Deliverables:** This BRD (functional and technical requirements), a high-level solution architecture integrating AI components, a working prototype (backend API + frontend interface), and supporting documentation for governance and future scale-up.

---

## 2. Stakeholder Requirements

Stakeholder needs from the project description are mapped to concrete requirements as follows.

| Need | Requirement ID | Requirement description |
|------|----------------|-------------------------|
| Identical names in different languages | FR-1 | The system shall support **multilingual text extraction and cross-language brand matching** (e.g. Hindi/Devanagari and English), including OCR, transliteration, translation, and semantic/phonetic similarity. |
| Replication of images with modified orientation/aspect | FR-2 | The system shall provide **rotation- and aspect-invariant visual similarity** via robust image embeddings (e.g. vision transformers, CNNs) and test-time augmentation (TTA) so that rotated or resized logos are still matched. |
| Synonyms or phonetically similar brand names | FR-3 | The system shall support **phonetic and semantic text similarity** (e.g. Soundex/Metaphone, multilingual sentence embeddings) so that synonyms and phonetically similar names are detected. |
| Visual or verbal mimicry causing consumer confusion | FR-4 | The system shall combine **multiple modalities** (visual embeddings, text, color, font, shape) in a single similarity score with **adaptive weighting and re-ranking** so that both visual and verbal mimicry are considered. |

---

## 3. Functional Requirements (FR)

| ID | Requirement | Traceability (e.g. API / component) |
|----|-------------|-------------------------------------|
| FR-1 | Multilingual text and cross-language brand matching | `text_engine.py`: PaddleOCR, EasyOCR, transliteration, translation, semantic + phonetic similarity; `api.py`: text score in similarity-check. |
| FR-2 | Rotation-invariant visual similarity | `engine.py`: DINOv2, VGG16; `api.py`: TTA variants (0°, 180°); `preprocessing.py`: rembg, pad-and-resize. |
| FR-3 | Phonetic and semantic text similarity | `text_engine.py`: sentence-transformers, jellyfish (Soundex/Metaphone), hybrid word similarity, stop-word and brand-match logic. |
| FR-4 | Multi-modal fusion and re-ranking | `fusion.py`: adaptive weights, RRF, z-score normalization; `api.py`: CandidateScores, rerank; modalities: DINO, VGG, text, color, font, shape. |
| FR-5 | Logo/image upload for similarity check | `POST /similarity-check`: accepts image file; returns top-K similar trademarks with per-modality scores. |
| FR-6 | Logo submission for repository indexing | `POST /submit-logo`: accepts image file; extracts all features and inserts into Milvus. |
| FR-7 | Batch repository indexing | `index_repo.py`: scans repository_images, extracts features, inserts into Milvus (batch); supports skip-if-exists. |
| FR-8 | Optional email notifications | `POST /send-email`: accepts receiver email, subject, HTML body; sends via Resend API. |
| FR-9 | Health/readiness check | `GET /`: returns service status and version. |

---

## 4. Non-Functional Requirements (NFR)

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Performance | Similarity-check response time should be acceptable for interactive use (e.g. target under ~15–30 seconds for a single query depending on hardware); batch indexing should support configurable batch size (e.g. 50 images per batch). |
| NFR-2 | Scalability | Vector storage and similarity search shall use a vector database (Milvus) with indexed fields for DINO, VGG, text, color, font, shape, icon; collection should support at least thousands of trademark images. |
| NFR-3 | Security | No API keys or secrets in source code; sensitive configuration (e.g. Resend API key) via environment variables; CORS configurable for frontend origin. |
| NFR-4 | Maintainability | Codebase modular (engine, text_engine, color_engine, shape_engine, fusion, preprocessing, api); dependencies pinned or versioned in requirements.txt. |

---

## 5. Technical Requirements / Solution Overview

### High-level stack
- **Backend:** Python 3.10, FastAPI, Uvicorn.
- **Vector DB:** Milvus (Lite), single collection with multiple vector and scalar fields.
- **ML/AI:** PyTorch, Transformers (DINOv2), TorchVision (VGG16), PaddleOCR, EasyOCR, sentence-transformers, rembg (U2-Net), scikit-learn, scikit-image, OpenCV, imagehash, jellyfish, indic-transliteration, deep-translator, pyemd.
- **Email:** Resend API (HTTP); sender configurable (e.g. onboarding@resend.dev or custom domain).
- **Frontend:** React, Vite, TypeScript, shadcn-ui, Tailwind (Lovable); communicates with backend API.
- **Deployment:** Docker; optional cloud deployment (e.g. AWS EC2).

### AI/ML building blocks (showcase)
- **Vision / deep learning:** DINOv2 (self-supervised vision transformer for logo embeddings), VGG16 (CNN for complementary visual features), U2-Net via rembg (background removal), TTA for rotation invariance; icon embedding; shape (Hu moments, contour histograms) and perceptual hashing (pHash) supporting retrieval.
- **NLP / text AI:** PaddleOCR and EasyOCR (multilingual OCR), sentence-transformers (multilingual text embeddings), phonetic matching (Soundex/Metaphone), transliteration (Devanagari→Latin), machine translation (Hindi/Marathi→English).
- **Fusion and ranking:** Z-score normalization, Reciprocal Rank Fusion (RRF), adaptive modality weighting, brand-match overrides (perfect text, soft text), pHash near-duplicate short-circuit.

The **high-level solution architecture** (system context, data flows, and placement of AI components) is described and illustrated in the accompanying Project Report.

---

## 6. Assumptions and Constraints

### Assumptions
- Repository images are available in a designated folder (`repository_images`) and can be batch-processed offline.
- Supported image formats: PNG, JPEG/JPG.
- Primary languages of interest: English, Hindi (Devanagari), and Marathi; OCR and translation are tuned accordingly.
- Email delivery for notifications is handled by a third-party provider (Resend); deliverability and domain verification are the responsibility of the deploying organization.
- Users of the similarity-check interface have access to the backend API (same host or CORS-enabled).

### Constraints
- Real-time video or live camera feed is out of scope.
- The system is designed for logo/trademark images (still images), not generic scene understanding.
- Milvus collection schema (e.g. trademark_v18) is fixed for a given deployment; schema changes require migration or re-indexing.

---

*End of BRD. For architecture diagrams and detailed implementation narrative, see the Project Report.*
