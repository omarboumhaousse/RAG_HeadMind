"""Step 5 : OCR — extract text from image regions using PaddleOCR 3.x.

Replaces Tesseract with PaddleOCR PP-OCRv5 for faster and more accurate
OCR on English, Chinese Simplified, and Chinese Traditional documents.

Engines used:
    - PaddleOCR (predict API)  : title, paragraph, caption, equation
    - PPStructureV3             : table  (returns structured Markdown)

Requires:
    paddleocr==3.0.3
    paddlepaddle==3.0.0
    paddlex==3.0.3
"""

import os
import numpy as np
import cv2

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"

# ── Singletons — chargés une seule fois ──────────────────────────
_OCR_ENGINE = None
_TABLE_ENGINE = None
_OCR_LOAD_FAILED = False
_TABLE_LOAD_FAILED = False


def _get_ocr_engine():
    """PaddleOCR pour le texte brut (title, paragraph, caption, equation)."""
    global _OCR_ENGINE, _OCR_LOAD_FAILED
    if _OCR_LOAD_FAILED:
        return None
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    try:
        from paddleocr import PaddleOCR
        _OCR_ENGINE = PaddleOCR(
            lang="ch",                          # Chinese + English
            device="cpu",
            use_doc_orientation_classify=False,  # inutile sur des crops pré-segmentés
            use_doc_unwarping=False,             # inutile sur des crops pré-segmentés
            use_textline_orientation=False,      # remplace use_angle_cls
        )
        print("[ocr] PaddleOCR engine loaded.")
        return _OCR_ENGINE
    except Exception as e:
        print(f"[ocr] PaddleOCR failed to load: {e}")
        _OCR_LOAD_FAILED = True
        return None


def _get_table_engine():
    """PPStructureV3 pour les blocs table — retourne du Markdown structuré."""
    global _TABLE_ENGINE, _TABLE_LOAD_FAILED
    if _TABLE_LOAD_FAILED:
        return None
    if _TABLE_ENGINE is not None:
        return _TABLE_ENGINE
    try:
        from paddleocr import PPStructureV3
        _TABLE_ENGINE = PPStructureV3(
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_table_recognition=True,
            use_formula_recognition=False,
            use_seal_recognition=False,
            use_chart_recognition=False,
        )
        print("[ocr] PPStructureV3 table engine loaded.")
        return _TABLE_ENGINE
    except Exception as e:
        print(f"[ocr] PPStructureV3 failed to load: {e}")
        _TABLE_LOAD_FAILED = True
        return None


# ── Preprocessing du crop ─────────────────────────────────────────

def _prepare_crop(crop: np.ndarray, label: str) -> np.ndarray:
    """
    Prépare le crop avant OCR :
    - Upscale les crops trop petits (caractères trop petits pour le recognizer)
    - Ajoute du padding pour que le détecteur ne rate pas les bords
    - Cap sur max_dim pour éviter les crops énormes (300 DPI)
    """
    h, w = crop.shape[:2]

    # Upscale si trop petit — seuil 48px hauteur pour CJK + Latin
    if h < 48:
        scale = max(2.0, 64 / max(h, 1))
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
        h, w = crop.shape[:2]

    # Cap sur 1500px côté le plus long — évite les crops géants de 300 DPI
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    # Padding léger — aide le détecteur sur les bords
    crop = cv2.copyMakeBorder(
        crop, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    return crop


# ── OCR texte brut ────────────────────────────────────────────────

def _ocr_text(crop: np.ndarray) -> str:
    engine = _get_ocr_engine()
    if engine is None:
        return ""

    try:
        results = list(engine.predict(input=crop))
    except Exception as e:
        print(f"[ocr] predict() failed: {e}")
        return ""

    if not results:
        return ""

    try:
        # OCRResult se comporte comme un dict — accès direct par clé
        texts  = results[0].get("rec_texts", [])
        scores = results[0].get("rec_scores", [])
    except Exception:
        return ""

    if not texts:
        return ""

    MIN_SCORE = 0.5
    kept = []
    for t, s in zip(texts, scores):
        t = t.strip()
        if t and float(s) >= MIN_SCORE:
            kept.append(t)

    return "\n".join(kept).strip()


# ── OCR table structurée ──────────────────────────────────────────

def _ocr_table(crop: np.ndarray) -> str:
    """
    Lance PPStructureV3 sur un crop de tableau.
    Essaie de retourner du Markdown structuré.
    Tombe en fallback sur _ocr_text() si PPStructureV3 échoue.
    """
    engine = _get_table_engine()
    if engine is None:
        # Fallback : OCR texte brut
        return _ocr_text(crop)

    try:
        import tempfile
        import json
        from pathlib import Path

        results = list(engine.predict(input=crop))
        if not results:
            return _ocr_text(crop)

        # PPStructureV3 peut sauvegarder en Markdown — on utilise un dossier temp
        with tempfile.TemporaryDirectory() as tmp_dir:
            results[0].save_to_markdown(save_path=tmp_dir)

            # Chercher le fichier .md généré
            md_files = list(Path(tmp_dir).glob("**/*.md"))
            if md_files:
                content = md_files[0].read_text(encoding="utf-8").strip()
                if content:
                    return content

        # Fallback : extraire rec_texts directement
        try:
            texts = results[0].get("rec_texts", [])
            return "\n".join(t.strip() for t in texts if t.strip())
        except Exception:
            return _ocr_text(crop)

    except Exception as e:
        print(f"[ocr] PPStructureV3 table OCR failed: {e}")
        return _ocr_text(crop)


# ── Point d'entrée public ─────────────────────────────────────────

def recognize_text(blocks: list[dict],
                   page_image: np.ndarray) -> list[dict]:
    """
    Extrait le texte de chaque bloc en utilisant PaddleOCR 3.x.

    Stratégie par label :
        figure             → "" (pas de texte)
        header / footer    → "" (ignoré pour le RAG)
        table              → PPStructureV3 (Markdown structuré)
        title / paragraph
        caption / equation → PaddleOCR plain text

    Args:
        blocks:     liste de blocs avec 'bbox' et 'label'
        page_image: image RGB de la page entière (numpy H×W×3 uint8)

    Returns:
        Même liste enrichie avec la clé 'text' sur chaque bloc.
    """
    h_img, w_img = page_image.shape[:2]

    # Labels à ignorer — pas de texte utile
    SKIP_LABELS = {"figure", "header", "footer"}

    for block in blocks:
        label = block.get("label", "paragraph")

        if label in SKIP_LABELS:
            block["text"] = ""
            continue

        # Crop de la région du bloc
        x, y, w, h = block["bbox"]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        crop = page_image[y1:y2, x1:x2]

        if crop.size == 0:
            block["text"] = ""
            continue

        # Preprocessing
        crop = _prepare_crop(crop, label)

        # Routing par label
        if label == "table":
            block["text"] = _ocr_table(crop)
        else:
            # title, paragraph, caption, equation
            block["text"] = _ocr_text(crop)

    return blocks