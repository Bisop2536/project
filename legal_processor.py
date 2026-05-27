import fitz  # PyMuPDF
import re
from typing import List, Dict, Any

class LegalDocProcessor:
    """
    Handles high-accuracy text extraction and layout analysis for legal PDF documents.
    """
    def __init__(self, header_footer_margin: float = 50.0):
        self.header_footer_margin = header_footer_margin

    def identify_legal_structure(self, text: str) -> Dict[str, Any]:
        """Identifies legal structures like section numbers, clauses, and hierarchy."""
        section_pattern = r'^((?:SECTION|Article)\s+\d+|(?:\d+\.)+\d*|\([a-z\d]\))\s+'
        match = re.match(section_pattern, text.strip(), re.IGNORECASE)
        structure = {"is_section_header": False, "section_number": None, "is_list_item": False}
        if match:
            structure["is_section_header"] = True
            structure["section_number"] = match.group(1)
        list_pattern = r'^\([ivx]+\)\s+'
        if re.match(list_pattern, text.strip(), re.IGNORECASE):
            structure["is_list_item"] = True
        return structure

    def analyze_layout(self, page: fitz.Page) -> Dict[str, Any]:
        """Analyzes the layout of a single page."""
        rect = page.rect
        return {
            "width": rect.width, "height": rect.height,
            "rotation": page.rotation, "is_landscape": rect.width > rect.height,
            "page_number": page.number + 1
        }

    def extract_tables(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Extracts tables from a page and converts them to Markdown text using pandas."""
        tables = []
        try:
            tabs = page.find_tables()
            for i, table in enumerate(tabs):
                df = table.to_pandas()
                if df.empty: continue
                table_text = df.to_markdown(index=False)
                tables.append({
                    "text": table_text, "bbox": table.bbox, "type": "table",
                    "page_number": page.number + 1, "table_index": i
                })
        except Exception: pass
        return tables

    def extract_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts text and tables while preserving reading order."""
        doc = fitz.open(pdf_path)
        extracted_content = []
        for page in doc:
            layout_info = self.analyze_layout(page)
            page_height = layout_info["height"]
            blocks = page.get_text("dict")["blocks"]
            tables = self.extract_tables(page)
            table_bboxes = [t["bbox"] for t in tables]
            page_elements = []

            for block in blocks:
                if "lines" not in block: continue
                bbox = block["bbox"]
                # Skip blocks that are part of an extracted table
                if any(bbox[0] >= t[0]-2 and bbox[1] >= t[1]-2 and bbox[2] <= t[2]+2 and bbox[3] <= t[3]+2 for t in table_bboxes):
                    continue

                text_content = "".join([span["text"] for line in block["lines"] for span in line["spans"]])
                if text_content.strip():
                    cleaned_text = text_content.strip()
                    page_elements.append({
                        "text": cleaned_text, "bbox": bbox, "type": "text",
                        "is_header": bbox[1] < self.header_footer_margin,
                        "is_footer": bbox[3] > page_height - self.header_footer_margin,
                        "page_number": layout_info["page_number"], "layout": layout_info,
                        "structure": self.identify_legal_structure(cleaned_text)
                    })
            for table in tables:
                table["layout"] = layout_info
                page_elements.append(table)
            # SORT elements by vertical position then horizontal position to preserve reading order
            page_elements.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
            extracted_content.extend(page_elements)
        doc.close()
        return extracted_content
