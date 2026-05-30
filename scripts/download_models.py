"""
Download PaddleOCR models to ~/.paddlex/official_models/.
Run once after installing dependencies:
    python scripts/download_models.py
"""
import os
import numpy as np

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"

dummy = np.zeros((100, 300, 3), dtype=np.uint8)

print("1/2 — Downloading PaddleOCR PP-OCRv5 (text)...")
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    lang="ch",
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
list(ocr.predict(input=dummy))
print("    OK — text OCR models downloaded")

print("\n2/2 — Downloading PPStructureV3 (tables)...")
from paddleocr import PPStructureV3
table = PPStructureV3(
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    use_table_recognition=True,
    use_formula_recognition=False,
    use_seal_recognition=False,
    use_chart_recognition=False,
)
list(table.predict(input=dummy))
print("    OK — PPStructureV3 models downloaded")

print("\nAll models saved to ~/.paddlex/official_models/")
