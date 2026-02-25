"""
text_engine.py
──────────────
Multilingual OCR (PaddleOCR + EasyOCR), text normalization,
Devanagari→Latin transliteration, Hindi/Marathi→English translation,
semantic text embeddings, hybrid text similarity (edit distance +
phonetic + semantic), stop word filtering, brand name boost, and
font visual embeddings via DINOv2 on cropped text regions.
"""

import os
import re
import json
import tempfile
import unicodedata
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

import cv2
import easyocr
import jellyfish
from PIL import Image
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
from deep_translator import GoogleTranslator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# ── globals (populated by initialize_text_engines) ────────────────────────

ocr_engines: dict = {}
easyocr_reader: easyocr.Reader | None = None
text_embed_model: SentenceTransformer | None = None

TEXT_EMBED_DIM = 384
FONT_EMBED_DIM = 768
MIN_WORD_LEN = 2
EDIT_SEMANTIC_ALPHA = 0.4
MIN_OCR_CONFIDENCE = 0.4

_SCRIPT_TO_LANG = {
    "DEVANAGARI": "hi",
    "BENGALI": "bn",
    "GUJARATI": "gu",
    "GURMUKHI": "pa",
    "KANNADA": "kn",
    "MALAYALAM": "ml",
    "TAMIL": "ta",
    "TELUGU": "te",
}

STOP_WORDS = frozenset({
    "the", "of", "in", "on", "at", "to", "for", "a", "an", "and",
    "or", "is", "it", "by", "with", "from", "as", "be", "was",
    "are", "been", "being", "this", "that", "these", "those",
    "ka", "ki", "ke", "se", "ko", "me", "hai", "ne",
    "aur", "ya", "par", "mein",
    "pvt", "ltd", "inc", "llc", "co", "corp",
})

_word_embed_cache: dict[str, np.ndarray] = {}


def _is_stop_word(w: str) -> bool:
    return w.lower().strip() in STOP_WORDS or len(w) <= 1


# =====================================================
#                 INITIALIZATION
# =====================================================

def initialize_text_engines():
    global ocr_engines, easyocr_reader, text_embed_model

    if text_embed_model is not None:
        return

    print("  ⏳ Loading PaddleOCR engines …")
    ocr_engines = {
        "en": PaddleOCR(use_angle_cls=True, lang="en", show_log=False),
        "hi": PaddleOCR(use_angle_cls=True, lang="hi", show_log=False),
        "mr": PaddleOCR(use_angle_cls=True, lang="mr", show_log=False),
    }

    print("  ⏳ Loading EasyOCR reader …")
    easyocr_reader = easyocr.Reader(["hi", "mr", "en"], gpu=False, verbose=False)

    print("  ⏳ Loading multilingual sentence-transformer …")
    text_embed_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("  ✅ Text engines ready.")


# =====================================================
#              UNICODE HELPERS
# =====================================================

def _normalize_text(raw: str) -> str:
    """NFKC normalize, strip punctuation/symbols/whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"[\s]+", "", text)
    text = re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    return text.casefold()


def _dominant_script(text: str) -> str:
    """Return 'LATIN', 'DEVANAGARI', etc. based on majority of characters."""
    counts: dict[str, int] = {}
    for ch in text:
        try:
            name = unicodedata.name(ch, "")
            script = name.split()[0] if name else "UNKNOWN"
        except ValueError:
            script = "UNKNOWN"
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "UNKNOWN"
    return max(counts, key=counts.get)


def _same_script(a: str, b: str) -> bool:
    return _dominant_script(a) == _dominant_script(b)


def _is_devanagari(text: str) -> bool:
    return _dominant_script(text) == "DEVANAGARI"


def _is_latin(text: str) -> bool:
    return _dominant_script(text) == "LATIN"


# =====================================================
#        TRANSLITERATION & TRANSLATION
# =====================================================

def _transliterate_devanagari(text: str) -> str:
    """Devanagari → IAST Latin transliteration, lowercased."""
    try:
        result = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        return result.lower()
    except Exception:
        return text.lower()


def _to_latin(text: str) -> str:
    """Transliterate to Latin if Devanagari, otherwise return lowercased."""
    if _is_devanagari(text):
        return _transliterate_devanagari(text)
    return text.lower()


def _translate_to_english(text: str, src_lang: str = "hi") -> str:
    """Translate a phrase to English using Google Translate (free)."""
    if not text.strip():
        return ""
    try:
        result = GoogleTranslator(source=src_lang, target="en").translate(text)
        return result.strip() if result else ""
    except Exception:
        return ""


def _detect_src_lang(text: str) -> str:
    """Map dominant script to a Google Translate language code."""
    script = _dominant_script(text)
    return _SCRIPT_TO_LANG.get(script, "hi")


# =====================================================
#          OCR IMAGE VARIANTS (for better OCR)
# =====================================================

def _ocr_image_variants(image_path: str) -> list[str]:
    """Generate contrast-enhanced, binarized, and inverted variants."""
    img = cv2.imread(image_path)
    if img is None:
        return [image_path]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paths = [image_path]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    p1 = tempfile.mktemp(suffix=".png")
    cv2.imwrite(p1, enhanced)
    paths.append(p1)

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    p2 = tempfile.mktemp(suffix=".png")
    cv2.imwrite(p2, binary)
    paths.append(p2)

    p3 = tempfile.mktemp(suffix=".png")
    cv2.imwrite(p3, cv2.bitwise_not(binary))
    paths.append(p3)

    return paths


# =====================================================
#              MULTILINGUAL OCR
# =====================================================

def _run_paddle_ocr(image_path: str) -> tuple[list[dict], list | None]:
    """
    Run all PaddleOCR engines, return the best result set
    (highest mean confidence) as a list of word-dicts and the raw result.
    """
    best_entries: list[dict] = []
    best_raw = None
    best_mean_conf = -1.0

    for lang, engine in ocr_engines.items():
        try:
            result = engine.ocr(image_path, cls=True)
        except Exception:
            continue

        entries: list[dict] = []
        confs: list[float] = []
        if result and result[0]:
            for line in result[0]:
                raw_txt = line[1][0]
                txt = _normalize_text(raw_txt)
                if len(txt) < MIN_WORD_LEN:
                    continue
                conf = float(line[1][1])
                if conf < MIN_OCR_CONFIDENCE:
                    continue
                entries.append({"word": txt, "raw": raw_txt, "conf": round(conf, 4)})
                confs.append(conf)

        mean_conf = float(np.mean(confs)) if confs else 0.0
        if mean_conf > best_mean_conf and entries:
            best_entries = entries
            best_mean_conf = mean_conf
            best_raw = result

    return best_entries, best_raw


def _run_easyocr(image_path: str) -> list[dict]:
    """Run EasyOCR and return word-dicts."""
    if easyocr_reader is None:
        return []
    try:
        results = easyocr_reader.readtext(image_path)
    except Exception:
        return []

    entries: list[dict] = []
    for bbox, raw_txt, conf in results:
        txt = _normalize_text(raw_txt)
        if len(txt) < MIN_WORD_LEN:
            continue
        conf = float(conf)
        if conf < MIN_OCR_CONFIDENCE:
            continue
        entries.append({"word": txt, "raw": raw_txt, "conf": round(conf, 4)})
    return entries


def _merge_ocr_entries(
    paddle_entries: list[dict], easy_entries: list[dict]
) -> list[dict]:
    """
    Union of OCR results, deduplicated by normalized word.
    When both engines detect the same word, keep the higher confidence.
    """
    merged: dict[str, dict] = {}
    for e in paddle_entries:
        w = e["word"]
        if w not in merged or e["conf"] > merged[w]["conf"]:
            merged[w] = e

    for e in easy_entries:
        w = e["word"]
        if w not in merged or e["conf"] > merged[w]["conf"]:
            merged[w] = e

    return list(merged.values())


def get_ocr_data(image_path: str) -> tuple[list[str], float, str, list | None]:
    """
    Run PaddleOCR + EasyOCR on original + image variants in parallel,
    merge results, then transliterate/translate any non-Latin words.

    Returns:
        words          – list of normalized words (original script)
        mean_conf      – mean OCR confidence
        ocr_json       – JSON string of [{word, raw, conf, transliterated?, translated?}, ...]
        raw_result     – raw PaddleOCR result (for bounding boxes / font cropping)
    """
    variants = _ocr_image_variants(image_path)

    try:
        def _paddle_all():
            all_entries: list[dict] = []
            orig_raw = None
            for v in variants:
                entries, raw = _run_paddle_ocr(v)
                all_entries.extend(entries)
                if v == image_path and raw is not None:
                    orig_raw = raw
            return all_entries, orig_raw

        def _easy_all():
            all_entries: list[dict] = []
            for v in variants:
                entries = _run_easyocr(v)
                all_entries.extend(entries)
            return all_entries

        with ThreadPoolExecutor(max_workers=2) as pool:
            paddle_future = pool.submit(_paddle_all)
            easy_future = pool.submit(_easy_all)
            paddle_entries, paddle_raw = paddle_future.result()
            easy_entries = easy_future.result()

    finally:
        for v in variants:
            if v != image_path and os.path.exists(v):
                try:
                    os.remove(v)
                except OSError:
                    pass

    merged = _merge_ocr_entries(paddle_entries, easy_entries)

    if not merged:
        return [], 0.0, json.dumps([], ensure_ascii=False), paddle_raw

    joined_non_latin = " ".join(
        e["raw"] for e in merged if _is_devanagari(e["word"])
    )
    full_translation = ""
    if joined_non_latin.strip():
        src_lang = _detect_src_lang(joined_non_latin)
        full_translation = _translate_to_english(joined_non_latin, src_lang)

    translated_words = full_translation.lower().split() if full_translation else []
    tw_idx = 0

    for entry in merged:
        w = entry["word"]
        if _is_devanagari(w):
            entry["transliterated"] = _transliterate_devanagari(w)
            if tw_idx < len(translated_words):
                entry["translated"] = translated_words[tw_idx]
                tw_idx += 1
            else:
                entry["translated"] = entry["transliterated"]
        else:
            entry["transliterated"] = w.lower()
            entry["translated"] = w.lower()

    words = [e["word"] for e in merged]
    confs = [e["conf"] for e in merged]
    mean_conf = float(np.mean(confs)) if confs else 0.0

    return (
        words,
        mean_conf,
        json.dumps(merged, ensure_ascii=False),
        paddle_raw,
    )


def extract_english_words(ocr_json: str) -> list[str]:
    """
    Pull the best English representation from each entry in ocr_json.
    Filters out stop words so text embeddings focus on meaningful content.
    """
    try:
        records = json.loads(ocr_json)
    except Exception:
        return []
    english: list[str] = []
    for r in records:
        w = r.get("translated") or r.get("transliterated") or r.get("word", "")
        w = w.strip()
        if w and not _is_stop_word(w):
            english.append(w)
    return english


# =====================================================
#     WORD EMBEDDING CACHE (batch precompute)
# =====================================================

def batch_precompute_embeddings(words: list[str]):
    """Encode all unique words in one shot. Call before scoring loop."""
    _word_embed_cache.clear()
    unique = list(set(w for w in words if w))
    if not unique or text_embed_model is None:
        return
    vecs = text_embed_model.encode(
        unique, normalize_embeddings=True, batch_size=128
    )
    for w, v in zip(unique, vecs):
        _word_embed_cache[w] = v


def clear_word_cache():
    _word_embed_cache.clear()


def _get_word_embedding(word: str) -> np.ndarray:
    if word in _word_embed_cache:
        return _word_embed_cache[word]
    if text_embed_model is None:
        return np.zeros(TEXT_EMBED_DIM, dtype=np.float32)
    vec = text_embed_model.encode(word, normalize_embeddings=True)
    _word_embed_cache[word] = vec
    return vec


# =====================================================
#           TEXT EMBEDDINGS (SEMANTIC)
# =====================================================

def get_text_embedding(words: list[str]) -> np.ndarray:
    """
    Encode a list of words into a single 384-dim embedding using
    the multilingual sentence-transformer.  Returns zero-vector if no words.
    """
    if not words or text_embed_model is None:
        return np.zeros(TEXT_EMBED_DIM, dtype=np.float32)

    joined = " ".join(words)
    vec = text_embed_model.encode(joined, normalize_embeddings=True)
    return vec.astype(np.float32)


# =====================================================
#           PHONETIC MATCHING
# =====================================================

def _phonetic_sim(a: str, b: str) -> float:
    """Metaphone + Soundex phonetic similarity for Latin-form words."""
    if not a or not b:
        return 0.0
    try:
        if jellyfish.metaphone(a) == jellyfish.metaphone(b):
            return 1.0
        if jellyfish.soundex(a) == jellyfish.soundex(b):
            return 0.85
    except Exception:
        pass
    return 0.0


# =====================================================
#           HYBRID TEXT SIMILARITY
# =====================================================

def _word_sim(q: str, r: str) -> float:
    """
    Hybrid similarity between two words using cached embeddings,
    edit distance, and phonetic matching.

    same-script  → max(alpha * edit_sim + (1-alpha) * semantic_sim, phonetic)
    cross-script → max(combined, semantic_sim, phonetic)
    """
    max_len = max(len(q), len(r))
    if max_len == 0:
        return 0.0

    q_emb = _get_word_embedding(q)
    r_emb = _get_word_embedding(r)
    semantic_sim = float(np.dot(q_emb, r_emb))

    q_lat = _to_latin(q)
    r_lat = _to_latin(r)

    if _same_script(q, r):
        edit_sim = 1.0 - levenshtein_distance(q, r) / max_len
        base = (
            EDIT_SEMANTIC_ALPHA * edit_sim
            + (1.0 - EDIT_SEMANTIC_ALPHA) * semantic_sim
        )
        phonetic = _phonetic_sim(q_lat, r_lat)
        return max(base, phonetic)

    lat_max_len = max(len(q_lat), len(r_lat), 1)
    edit_sim_lat = 1.0 - levenshtein_distance(q_lat, r_lat) / lat_max_len
    combined = (
        EDIT_SEMANTIC_ALPHA * edit_sim_lat
        + (1.0 - EDIT_SEMANTIC_ALPHA) * semantic_sim
    )
    phonetic = _phonetic_sim(q_lat, r_lat)
    return max(combined, semantic_sim, phonetic)


def calculate_text_score(query_words: list[str], ocr_json: str) -> float:
    """
    Symmetric best-match aggregation with stop word filtering and
    brand name match boost.

    Stop words are filtered from both sides before matching.
    If ANY single non-stop-word pair matches >= 0.95, the score is
    floored at 0.9 (brand name match boost).
    """
    try:
        records = json.loads(ocr_json)
    except Exception:
        return 0.0

    if not query_words or not records:
        return 0.0

    filtered_query = [w for w in query_words if not _is_stop_word(w)]
    if not filtered_query:
        filtered_query = query_words

    repo_forms: list[list[str]] = []
    for r in records:
        forms = []
        for key in ("word", "transliterated", "translated"):
            v = r.get(key)
            if v and not _is_stop_word(v):
                forms.append(v)
        if forms:
            repo_forms.append(forms)

    if not repo_forms:
        for r in records:
            forms = [r["word"]]
            if r.get("transliterated"):
                forms.append(r["transliterated"])
            if r.get("translated"):
                forms.append(r["translated"])
            repo_forms.append(forms)

    best_single = 0.0

    def _best_match_q_to_r(qw: str) -> float:
        nonlocal best_single
        best = 0.0
        for forms in repo_forms:
            for form in forms:
                s = _word_sim(qw, form)
                if s > best:
                    best = s
                if s > best_single:
                    best_single = s
        return best

    def _best_match_r_to_q(repo_entry_forms: list[str]) -> float:
        nonlocal best_single
        best = 0.0
        for form in repo_entry_forms:
            for qw in filtered_query:
                s = _word_sim(form, qw)
                if s > best:
                    best = s
                if s > best_single:
                    best_single = s
        return best

    q_to_r = [_best_match_q_to_r(qw) for qw in filtered_query]
    r_to_q = [_best_match_r_to_q(forms) for forms in repo_forms]

    score = float(0.5 * np.mean(q_to_r) + 0.5 * np.mean(r_to_q))

    if best_single >= 0.95:
        score = max(score, 0.9)

    return score


# =====================================================
#            FONT / TYPOGRAPHY EMBEDDING
# =====================================================

def _pad_to_square(img: Image.Image, fill=(128, 128, 128)) -> Image.Image:
    w, h = img.size
    size = max(w, h)
    padded = Image.new("RGB", (size, size), fill)
    padded.paste(img, ((size - w) // 2, (size - h) // 2))
    return padded


def crop_text_regions(image: Image.Image, raw_ocr_result) -> list[Image.Image]:
    """Crop text bounding boxes from the original image."""
    regions = []
    if not raw_ocr_result or not raw_ocr_result[0]:
        return regions
    for line in raw_ocr_result[0]:
        bbox = line[0]
        x_min = int(min(p[0] for p in bbox))
        y_min = int(min(p[1] for p in bbox))
        x_max = int(max(p[0] for p in bbox))
        y_max = int(max(p[1] for p in bbox))
        if x_max <= x_min or y_max <= y_min:
            continue
        region = image.crop((x_min, y_min, x_max, y_max)).convert("RGB")
        regions.append(region)
    return regions


@torch.inference_mode()
def get_font_embedding(
    text_regions: list[Image.Image],
    dino_processor,
    dino_model,
    device: str = "cpu",
) -> np.ndarray:
    """
    Average DINOv2 CLS embedding over cropped text regions.
    Captures font weight, style, stroke — independent of text content.
    Returns zero-vector if no text regions.
    """
    if not text_regions:
        return np.zeros(FONT_EMBED_DIM, dtype=np.float32)

    padded = [
        _pad_to_square(r).resize((224, 224))
        for r in text_regions
    ]
    inputs = dino_processor(images=padded, return_tensors="pt").to(device)
    vecs = dino_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    vecs = vecs / norms

    mean_vec = vecs.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 1e-10:
        mean_vec /= norm
    return mean_vec.astype(np.float32)
