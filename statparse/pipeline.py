"""End-to-end StatParse pipeline."""
from pathlib import Path

from .pdf_to_image import render_pdf #librairie
from .preprocessing import preprocess #Omar + Edgar
from .segmentation import segment #Timothée
from .classification import classify #todo
from .reading_order import order_blocks #todo
from .ocr import recognize_text #tesseract 
from .serialization import to_markdown #librairie 


class Pipeline:
    """StatParse document parsing pipeline.

    Usage:
        pipeline = Pipeline(dpi=300)
        pages_md = pipeline.parse("document.pdf")
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def parse(self, pdf_path: str) -> list[str]:
        """Parse a PDF and return a list of markdown strings (one per page)."""
        images = render_pdf(pdf_path, dpi=self.dpi)
        results = []

        for page_image in images:
            clean = preprocess(page_image)
            blocks = segment(clean)
            labeled = classify(blocks)
            ordered = order_blocks(labeled)
            text_blocks = recognize_text(ordered, page_image)
            md = to_markdown(text_blocks)
            results.append(md)

        return results

    def parse_to_file(self, pdf_path: str, output_path: str) -> None:
        """Parse a PDF and write the combined markdown to a file."""
        pages = self.parse(pdf_path)
        combined = "\n\n---\n\n".join(pages)
        Path(output_path).write_text(combined, encoding="utf-8")
