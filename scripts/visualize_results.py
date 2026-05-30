# visualize_results.py
"""
Produit des visualisations claires pour présentation.

Pour chaque annotation :
    1. Classification : manuel vs auto, couleurs par label, légende
    2. Reading order  : numéros d'ordre sur chaque box
    3. OCR preview    : texte extrait affiché sur chaque box

Usage :
    python visualize_results.py
"""

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"

import json
import textwrap
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

from statparse.classification import classify
from statparse.reading_order import order_blocks
from statparse.ocr import recognize_text
from statparse.serialization import to_markdown

# ── Configuration ─────────────────────────────────────────────────
ANNOT_DIR  = Path("./annotations")
OUTPUT_DIR = Path("./result/presentation2")
DPI        = 150

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette de couleurs par label (RGB) ──────────────────────────
PALETTE = {
    "title":     (220,  50,  50),
    "paragraph": ( 34, 160,  34),
    "table":     ( 50,  50, 220),
    "figure":    (220, 130,   0),
    "caption":   (150,   0, 200),
    "header":    (  0, 180, 180),
    "footer":    (180, 140,   0),
    "equation":  (220,  20, 140),
    "unknown":   (150, 150, 150),
}

LABEL_FR = {
    "title":     "Titre",
    "paragraph": "Paragraphe",
    "table":     "Tableau",
    "figure":    "Figure",
    "caption":   "Légende",
    "header":    "En-tête",
    "footer":    "Pied de page",
    "equation":  "Équation",
    "unknown":   "Inconnu",
}


def load_annotation(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for block in data["blocks"]:
        block["bbox"] = tuple(block["bbox"])
        for comp in block.get("components", []):
            comp["bbox"] = tuple(comp["bbox"])
    return data


def get_pdf_image(data: dict) -> np.ndarray:
    # Prendre uniquement le nom du fichier, ignorer le chemin complet Windows
    pdf_name = Path(data["pdf"].replace("\\", "/")).name
    pdf_path = Path("./pdfs") / pdf_name
    pages = convert_from_path(str(pdf_path), dpi=DPI)
    return np.array(pages[0])


def draw_box_with_label(draw: ImageDraw.Draw,
                         bbox: tuple,
                         label: str,
                         color: tuple,
                         prefix: str = "",
                         thickness: int = 3,
                         font_size: int = 22):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    # Rectangle
    for t in range(thickness):
        draw.rectangle([x-t, y-t, x2+t, y2+t], outline=color)

    # Fond du texte
    text = f"{prefix}{label}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                   font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox_text = draw.textbbox((x+4, max(y-font_size-6, 2)), text, font=font)
    pad = 3
    draw.rectangle([bbox_text[0]-pad, bbox_text[1]-pad,
                    bbox_text[2]+pad, bbox_text[3]+pad],
                   fill=color)
    draw.text((x+4, max(y-font_size-6, 2)), text,
              fill=(255, 255, 255), font=font)


def draw_order_number(draw: ImageDraw.Draw,
                       bbox: tuple,
                       number: int,
                       label: str,
                       color: tuple):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    cx, cy = x + w // 2, y + h // 2

    # Rectangle de la box
    for t in range(3):
        draw.rectangle([x-t, y-t, x2+t, y2+t], outline=color)

    # Grand cercle avec le numéro
    r = 28
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)

    try:
        font_num = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        font_lbl = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font_num = ImageFont.load_default()
        font_lbl = font_num

    num_text = str(number)
    bbox_num = draw.textbbox((0, 0), num_text, font=font_num)
    tw = bbox_num[2] - bbox_num[0]
    th = bbox_num[3] - bbox_num[1]
    draw.text((cx - tw//2, cy - th//2), num_text,
              fill=(255, 255, 255), font=font_num)

    # Label en bas de la box
    lbl_text = LABEL_FR.get(label, label)
    bbox_lbl = draw.textbbox((0, 0), lbl_text, font=font_lbl)
    lw = bbox_lbl[2] - bbox_lbl[0]
    lx = max(x, cx - lw // 2)
    draw.rectangle([lx-2, y2+2, lx+lw+2, y2+22], fill=color)
    draw.text((lx, y2+2), lbl_text, fill=(255,255,255), font=font_lbl)


def draw_ocr_text(draw: ImageDraw.Draw,
                   bbox: tuple,
                   text: str,
                   label: str,
                   color: tuple):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    # Rectangle
    for t in range(3):
        draw.rectangle([x-t, y-t, x2+t, y2+t], outline=color)

    if not text.strip():
        # Indiquer que le texte est vide
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x+6, y+6), "(vide)", fill=(180, 180, 180), font=font)
        return

    # Afficher les 2 premières lignes du texte extrait
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except Exception:
        font = ImageFont.load_default()

    preview = text.strip().replace("\n", " ")[:120]
    wrapped = textwrap.wrap(preview, width=max(10, w // 11))[:3]

    bg_h = len(wrapped) * 22 + 8
    draw.rectangle([x+2, y+2, x2-2, y+2+bg_h],
                   fill=(255, 255, 255, 200))

    for i, line in enumerate(wrapped):
        draw.text((x+6, y+6 + i*22), line, fill=(30, 30, 30), font=font)


def add_legend(image: np.ndarray, title: str) -> np.ndarray:
    """Ajoute une bande de légende en bas de l'image."""
    legend_h = 110
    h, w = image.shape[:2]
    legend = np.ones((legend_h, w, 3), dtype=np.uint8) * 245

    pil = Image.fromarray(legend)
    draw = ImageDraw.Draw(pil)

    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_label = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = font_title

    draw.text((10, 6), title, fill=(30, 30, 30), font=font_title)

    labels = list(PALETTE.keys())[:-1]
    x_pos = 10
    for lbl in labels:
        color = PALETTE[lbl]
        draw.rectangle([x_pos, 36, x_pos+18, 54], fill=color)
        draw.text((x_pos+22, 36), LABEL_FR.get(lbl, lbl),
                  fill=(30, 30, 30), font=font_label)
        x_pos += len(LABEL_FR.get(lbl, lbl)) * 10 + 40
        if x_pos > w - 150:
            x_pos = 10

    legend_arr = np.array(pil)
    return np.vstack([image, legend_arr])


def process(annot_path: Path):
    data  = load_annotation(annot_path)
    stem  = data["stem"]
    print(f"\n{'='*55}")
    print(f"  {stem}")
    print(f"{'='*55}")

    page_image = get_pdf_image(data)
    blocks     = data["blocks"]
    page_shape = tuple(data["page_shape"][:2])

    # ── Classification ────────────────────────────────────────────
    classified = classify(blocks, image_shape=page_shape)

    # Image 1A — Labels manuels
    img_manual = Image.fromarray(page_image.copy())
    draw = ImageDraw.Draw(img_manual)
    for b in classified:
        manual = b.get("_manual_label", "unknown")
        color  = PALETTE.get(manual, PALETTE["unknown"])
        draw_box_with_label(draw, b["bbox"], manual, color, prefix="")
    img_manual = add_legend(np.array(img_manual),
                             "Annotation manuelle (vérité terrain)")

    # Image 1B — Labels automatiques avec ✓/✗
    img_auto = Image.fromarray(page_image.copy())
    draw = ImageDraw.Draw(img_auto)
    correct = 0
    for b in classified:
        manual = b.get("_manual_label", "unknown")
        auto   = b.get("label", "unknown")
        color  = PALETTE.get(auto, PALETTE["unknown"])
        ok     = "✓ " if manual == auto else "✗ "
        draw_box_with_label(draw, b["bbox"], auto, color, prefix=ok)
        if manual == auto:
            correct += 1
    total    = len(classified)
    accuracy = correct / total * 100 if total > 0 else 0
    img_auto = add_legend(
        np.array(img_auto),
        f"Classification automatique — Précision : {correct}/{total} ({accuracy:.0f}%)"
    )

    # Sauvegarder côte à côte
    h1 = img_manual.shape[0]
    h2 = img_auto.shape[0]
    max_h = max(h1, h2)
    w  = img_manual.shape[1]

    def pad_h(img, target_h):
        if img.shape[0] < target_h:
            pad = np.ones((target_h - img.shape[0], img.shape[1], 3),
                          dtype=np.uint8) * 245
            return np.vstack([img, pad])
        return img

    combined_classif = np.hstack([pad_h(img_manual, max_h),
                                   pad_h(img_auto,   max_h)])
    Image.fromarray(combined_classif).save(
        OUTPUT_DIR / f"{stem}_1_classification.png")
    print(f"  ✓ Classification sauvegardée  ({accuracy:.0f}% correct)")

    # ── Reading order ─────────────────────────────────────────────
    ordered = order_blocks(classified)

    img_ord = Image.fromarray(page_image.copy())
    draw    = ImageDraw.Draw(img_ord)
    for i, b in enumerate(ordered, 1):
        label = b.get("label", "unknown")
        color = PALETTE.get(label, PALETTE["unknown"])
        draw_order_number(draw, b["bbox"], i, label, color)

    img_ord_arr = add_legend(np.array(img_ord), "Ordre de lecture")
    Image.fromarray(img_ord_arr).save(
        OUTPUT_DIR / f"{stem}_2_reading_order.png")
    print(f"  ✓ Reading order sauvegardé")

    # ── OCR ───────────────────────────────────────────────────────
    text_blocks = recognize_text(ordered, page_image)

    img_ocr = Image.fromarray(page_image.copy())
    draw    = ImageDraw.Draw(img_ocr)
    for b in text_blocks:
        label = b.get("label", "unknown")
        text  = b.get("text", "")
        color = PALETTE.get(label, PALETTE["unknown"])
        draw_ocr_text(draw, b["bbox"], text, label, color)

    img_ocr_arr = add_legend(np.array(img_ocr), "Texte extrait par OCR (PaddleOCR PP-OCRv5)")
    Image.fromarray(img_ocr_arr).save(
        OUTPUT_DIR / f"{stem}_3_ocr.png")
    print(f"  ✓ OCR sauvegardée")

    # ── Markdown final ────────────────────────────────────────────
    md = to_markdown(text_blocks)
    md_path = OUTPUT_DIR / f"{stem}.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  ✓ Markdown sauvegardé ({len(md)} caractères)")

    # ── Résumé classification dans le terminal ────────────────────
    print(f"\n  Détail classification :")
    print(f"  {'BBOX':<25} {'MANUEL':<14} {'AUTO':<14} {'OK?'}")
    print(f"  {'-'*60}")
    for b in classified:
        manual = b.get("_manual_label", "?")
        auto   = b.get("label", "?")
        ok     = "✓" if manual == auto else "✗"
        x, y, w, h = b["bbox"]
        print(f"  ({x:4d},{y:4d},{w:4d},{h:4d})  {manual:<14} {auto:<14} {ok}")


if __name__ == "__main__":
    annots = sorted(ANNOT_DIR.glob("*.json"))
    if not annots:
        print(f"Aucune annotation dans {ANNOT_DIR}/")
        exit(1)

    print(f"\nAnnotations : {len(annots)}")
    print(f"Sortie      : {OUTPUT_DIR}/")

    total_correct = 0
    total_blocks  = 0

    for annot in annots:
        try:
            process(annot)
        except Exception as e:
            print(f"ERREUR sur {annot.name} : {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"Terminé. Images dans : {OUTPUT_DIR}/")
    print(f"{'='*55}")