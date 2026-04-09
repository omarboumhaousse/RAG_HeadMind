"""End-to-end StatParse pipeline."""
from pathlib import Path
import numpy as np

from .pdf_to_image import render_pdf
from .preprocessing import preprocess
from .segmentation import segment
from .classification import classify
from .reading_order import order_blocks
from .ocr import recognize_text
from .serialization import to_markdown


class Pipeline:
    """StatParse document parsing pipeline.

    Usage:
        pipeline = Pipeline(dpi=150)
        pages_md = pipeline.parse("document.pdf")
    """

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def parse(self, pdf_path: str) -> list[str]:
        """Parse a PDF and return one Markdown string per page."""
        images = render_pdf(pdf_path, dpi=self.dpi)
        results = []
        for page_image in images:
            results.append(self.parse_image(page_image))
        return results

    def parse_image(self, image: np.ndarray) -> str:
        """Parse a single page image and return Markdown."""
        clean       = preprocess(image)
        blocks      = segment(clean)
        labeled     = classify(blocks, image_shape=clean.shape)
        ordered     = order_blocks(labeled, image_shape=clean.shape)
        text_blocks = recognize_text(ordered, image)
        return to_markdown(text_blocks)

    def parse_to_file(self, pdf_path: str, output_path: str) -> None:
        """Parse a PDF and write combined Markdown to a file."""
        pages    = self.parse(pdf_path)
        combined = "\n\n---\n\n".join(pages)
        Path(output_path).write_text(combined, encoding="utf-8")