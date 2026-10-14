"""
shared/ocr_reader.py
----------------------------------------------------------------------------
High-performance OCR Singleton shared across spine, scattered, and mixed
pipelines.

Primary Engine:  RapidOCR (PP-OCRv4 ONNX) — lightning fast (~15ms), handles arbitrary
                 orientations, accurate Latin & multilingual reading without character hallucination.
Fallback Engine: EasyOCR ('en') — robust backup if ONNX engine is unavailable.
"""

import sys
import numpy as np

_shared_ocr_reader = None


class UnifiedOCR:
    """
    Unified high-performance OCR wrapper providing a standardized
    readtext(img, detail=0) interface.
    """
    def __init__(self, verbose=True):
        self._rapid_engine = None
        self._easy_reader = None
        self.engine_name = "rapidocr"

        try:
            from rapidocr_onnxruntime import RapidOCR
            self._rapid_engine = RapidOCR(intra_op_num_threads=4)
            if verbose:
                print("[Shared OCR] RapidOCR engine ready (PP-OCRv4 ONNX).", flush=True)
        except Exception as e:
            if verbose:
                print(f"[Shared OCR] RapidOCR unavailable ({e}). Initializing EasyOCR fallback.", flush=True)
            self._init_easyocr(verbose=verbose)

    def _init_easyocr(self, languages=("en",), verbose=False):
        if self._easy_reader is None:
            import torch
            import easyocr
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu = (device == "cuda")
            if verbose:
                print(f"[Shared OCR] Loading EasyOCR fallback ({list(languages)}, gpu={gpu})...", flush=True)
            self._easy_reader = easyocr.Reader(list(languages), gpu=gpu)
            if verbose:
                print("[Shared OCR] EasyOCR fallback ready.", flush=True)
            self.engine_name = "easyocr"

    def readtext(self, img, detail=0, **kwargs):
        """
        Standardized readtext interface matching EasyOCR API.
        Returns:
            detail=0 -> list of detected strings
            detail=1 -> list of (box, text, score)
        """
        if img is None:
            return []

        # 1. Primary: RapidOCR (PP-OCRv4 ONNX)
        if self._rapid_engine is not None:
            try:
                res, _ = self._rapid_engine(img)
                if not res and isinstance(img, np.ndarray) and len(img.shape) == 3 and img.size > 0:
                    # Light CLAHE contrast pass for low-contrast or glare crops
                    import cv2
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
                    res, _ = self._rapid_engine(enhanced)

                if res:
                    if detail == 0:
                        return [line[1] for line in res]
                    return [(line[0], line[1], float(line[2])) for line in res]
                return []
            except Exception:
                pass

        # 2. Fallback: EasyOCR
        if self._easy_reader is None:
            self._init_easyocr()
        return self._easy_reader.readtext(img, detail=detail, **kwargs)


def clean_tokens(text):
    """Normalizes text by stripping punctuation and splitting into lower-case words."""
    import re
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) > 0]


COMMON_STOPWORDS = {
    "of", "the", "a", "an", "and", "in", "on", "for", "with", "to",
    "at", "by", "from", "edition", "vol", "volume", "da", "de", "la"
}


_catalog_cache = None


def _get_catalog():
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache
    from pathlib import Path
    import json
    p1 = Path(__file__).resolve().parent.parent / "mock_library_catalog.json"
    p2 = Path(__file__).resolve().parent.parent / "library_catalog.json"
    path = p1 if p1.exists() else (p2 if p2.exists() else None)
    if path and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                _catalog_cache = json.load(f)
        except Exception:
            _catalog_cache = []
    else:
        _catalog_cache = []
    return _catalog_cache


def _find_catalog_book(query):
    if not query:
        return None
    if isinstance(query, dict):
        return query
    catalog = _get_catalog()
    q_low = str(query).lower().strip()
    # Check exact book_id or title
    for b in catalog:
        if q_low == b.get("book_id", "").lower():
            return b
        if q_low == b.get("title", "").lower():
            return b
        for a in b.get("aliases", []):
            if q_low == a.lower():
                return b
    # Check substring containment
    for b in catalog:
        if q_low in b.get("title", "").lower() or b.get("title", "").lower() in q_low:
            return b
        if q_low in b.get("author", "").lower() or b.get("author", "").lower() in q_low:
            return b
    return None


def compute_title_similarity(ocr_text, query_title, target_book=None):
    """
    Robust title similarity matching supporting English & Chinese:
    1. Exact full match or bidirectional substring containment -> 1.0
    2. Catalog-aware multi-field matching: leverages Title + Author + Aliases + Edition
    3. Author disambiguation: boosts matching author, penalizes competing authors (e.g. 3 Linear Algebra books)
    4. Stopword-aware token matching + SequenceMatcher typo tolerance
    5. False-positive suppression for short letter noise
    """
    if not ocr_text or not query_title:
        return 0.0

    text_lower = str(ocr_text).lower().strip()

    # Extract target info (from dict or string query)
    if isinstance(query_title, dict):
        target_book = query_title
        query_str = target_book.get("title", "")
    else:
        query_str = str(query_title)

    if target_book is None:
        target_book = _find_catalog_book(query_str)

    query_lower = query_str.lower().strip()

    # Exact or full substring containment
    if query_lower in text_lower or text_lower in query_lower:
        return 1.0

    # Chinese direct character matching (e.g., 線性代數)
    if target_book and "aliases" in target_book:
        for alias in target_book["aliases"]:
            a_low = alias.lower().strip()
            if any("\u4e00" <= ch <= "\u9fff" for ch in a_low):
                if a_low in text_lower or any(part in text_lower for part in a_low.split() if len(part) >= 2):
                    return 0.95

    from difflib import SequenceMatcher

    q_tokens = clean_tokens(query_lower)
    t_tokens = clean_tokens(text_lower)

    if not q_tokens or not t_tokens:
        r = SequenceMatcher(None, text_lower, query_lower).ratio()
        return r if r >= 0.65 else 0.0

    # Token-set exact subset match
    q_set = set(q_tokens)
    t_set = set(t_tokens)
    if q_set.issubset(t_set) or t_set.issubset(q_set):
        return 1.0

    # Separate significant tokens (words with len >= 3 not in stopwords)
    q_sig = [w for w in q_tokens if w not in COMMON_STOPWORDS and len(w) >= 3]
    t_sig = [w for w in t_tokens if w not in COMMON_STOPWORDS and len(w) >= 3]

    if not q_sig:
        q_sig = q_tokens
    if not t_sig:
        t_sig = t_tokens

    # Significant token subset check
    if set(q_sig).issubset(set(t_sig)) or set(t_sig).issubset(set(q_sig)):
        return 1.0

    # Match each significant query token against OCR tokens
    sig_scores = []
    for qw in q_sig:
        best_w_score = 0.0
        for tw in t_tokens:
            if qw == tw:
                best_w_score = 1.0
                break
            # Substring match for concatenated or partially recognized words (min 4 chars)
            if len(qw) >= 4 and len(tw) >= 4:
                if qw in tw or tw in qw:
                    sub_ratio = min(len(qw), len(tw)) / max(len(qw), len(tw))
                    if sub_ratio >= 0.60 and sub_ratio > best_w_score:
                        best_w_score = sub_ratio
            # SequenceMatcher ratio for typos (e.g. concepts -> onceptsoe, bretschar -> bretscher)
            r = SequenceMatcher(None, qw, tw).ratio()
            if r > best_w_score:
                best_w_score = r

        # Only accept strong token matches (>= 0.55) to prevent noise accumulation
        sig_scores.append(best_w_score if best_w_score >= 0.55 else 0.0)

    # Evaluate match quality across significant tokens
    sig_matched = [s for s in sig_scores if s >= 0.55]
    if len(sig_matched) == len(q_sig) and len(q_sig) > 0:
        # All significant words matched!
        mean_score = sum(sig_scores) / len(sig_scores)
        fuzzy_score = min(1.0, max(0.90, mean_score))
    elif sig_matched:
        # Partial match: scale by proportion of matched significant words
        coverage = len(sig_matched) / len(q_sig)
        mean_score = sum(sig_scores) / len(q_sig)
        fuzzy_score = mean_score * coverage
    else:
        fuzzy_score = 0.0

    # Catalog author and alias reinforcement
    author_boost = 0.0
    if target_book:
        author_name = target_book.get("author", "").lower()
        author_tokens = clean_tokens(author_name)
        author_sig = [w for w in author_tokens if w not in COMMON_STOPWORDS and len(w) >= 3]

        for aw in author_sig:
            for tw in t_tokens:
                if aw == tw or (len(aw) >= 4 and len(tw) >= 4 and (aw in tw or tw in aw)):
                    author_boost = max(author_boost, 0.40)
                    break
                r = SequenceMatcher(None, aw, tw).ratio()
                if r >= 0.75:
                    author_boost = max(author_boost, 0.35)
                    break

        # Check aliases
        for alias in target_book.get("aliases", []):
            al_low = alias.lower()
            if al_low in text_lower or (len(al_low) >= 4 and al_low in text_lower):
                author_boost = max(author_boost, 0.45)
                break

    # If significant title words matched and author also confirmed, boost to high confidence!
    if fuzzy_score >= 0.35 and author_boost > 0.0:
        fuzzy_score = min(1.0, max(0.85, fuzzy_score + author_boost))
    elif author_boost >= 0.40 and fuzzy_score < 0.35:
        # Author alone found with high confidence
        fuzzy_score = max(fuzzy_score, author_boost)

    # Sliding window substring ratio for dense text blocks
    window_size = len(query_lower)
    best_window = 0.0
    if len(text_lower) >= window_size:
        for i in range(len(text_lower) - window_size + 1):
            chunk = text_lower[i:i + window_size]
            r = SequenceMatcher(None, chunk, query_lower).ratio()
            if r > best_window:
                best_window = r
    if best_window < 0.65:
        best_window = 0.0

    full_ratio = SequenceMatcher(None, text_lower, query_lower).ratio()
    if full_ratio < 0.65:
        full_ratio = 0.0

    return max(full_ratio, fuzzy_score, best_window)


# Shared alias
title_similarity = compute_title_similarity


def get_shared_ocr_reader(languages=("en",), gpu=None, verbose=True):
    """Lazy-load the shared OCR singleton once, reused across all pipelines."""
    global _shared_ocr_reader
    if _shared_ocr_reader is None:
        _shared_ocr_reader = UnifiedOCR(verbose=verbose)
    return _shared_ocr_reader


def preload_shared_ocr_reader(languages=("en",), gpu=None, verbose=False):
    """Force-load the shared OCR reader immediately and warm up ONNX graph in background."""
    reader = get_shared_ocr_reader(languages=languages, gpu=gpu, verbose=verbose)
    try:
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        reader.readtext(dummy)
    except Exception:
        pass

