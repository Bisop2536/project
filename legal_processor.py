import fitz  # PyMuPDF
import re
import numpy as np
from typing import List, Dict, Any, Optional

class LegalDocProcessor:
    """
    Handles high-accuracy text extraction and layout analysis for legal PDF documents.
    """
    def __init__(self, header_footer_margin: float = 50.0):
        self.header_footer_margin = header_footer_margin

    def identify_legal_structure(self, text: str) -> Dict[str, Any]:
        """
        Identifies legal structures such as section numbers, clauses, and hierarchy.
        """
        # Example patterns for section numbers: "1.", "1.1", "SECTION 1", "(a)", "Article 1"
        section_pattern = r'^((?:SECTION|Article)\s+\d+|(?:\d+\.)+\d*|\([a-z\d]\))\s+'
        match = re.match(section_pattern, text.strip(), re.IGNORECASE)

        structure = {
            "is_section_header": False,
            "section_number": None,
            "is_list_item": False
        }

        if match:
            structure["is_section_header"] = True
            structure["section_number"] = match.group(1)

        # Check for list items like (i), (ii), etc.
        list_pattern = r'^\([ivx]+\)\s+'
        if re.match(list_pattern, text.strip(), re.IGNORECASE):
            structure["is_list_item"] = True

        return structure

    def analyze_layout(self, page: fitz.Page) -> Dict[str, Any]:
        """
        Analyzes the layout of a single page.
        """
        rect = page.rect
        rotation = page.rotation
        is_landscape = rect.width > rect.height

        return {
            "width": rect.width,
            "height": rect.height,
            "rotation": rotation,
            "is_landscape": is_landscape,
            "page_number": page.number + 1
        }

    def extract_tables(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Extracts tables from a page and converts them to structured text.
        """
        tables = []
        try:
            tabs = page.find_tables()
            for i, table in enumerate(tabs):
                df = table.to_pandas()
                if df.empty:
                    continue

                # Convert table to a structured string representation (Markdown-like)
                table_text = df.to_markdown(index=False)

                tables.append({
                    "text": table_text,
                    "bbox": table.bbox,
                    "type": "table",
                    "page_number": page.number + 1,
                    "table_index": i
                })
        except Exception as e:
            # Table extraction can be brittle, handle gracefully
            print(f"Warning: Table extraction failed on page {page.number + 1}: {e}")

        return tables

    def extract_text_with_layout(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text and tables from a PDF while preserving layout information
         and natural reading order.
        """
        doc = fitz.open(pdf_path)
        extracted_content = []

        for page in doc:
            layout_info = self.analyze_layout(page)
            page_height = layout_info["height"]

            # Get text blocks with detailed information
            blocks = page.get_text("dict")["blocks"]

            # Extract tables
            tables = self.extract_tables(page)
            table_bboxes = [t["bbox"] for t in tables]

            page_elements = []

            for block in blocks:
                if "lines" not in block:
                    continue

                bbox = block["bbox"]

                # Check if this block is inside a table we already extracted
                is_inside_table = False
                for t_bbox in table_bboxes:
                    # Simple intersection check
                    if (bbox[0] >= t_bbox[0] - 2 and bbox[1] >= t_bbox[1] - 2 and
                        bbox[2] <= t_bbox[2] + 2 and bbox[3] <= t_bbox[3] + 2):
                        is_inside_table = True
                        break

                if is_inside_table:
                    continue

                # Identify if block is likely header or footer
                is_header = bbox[1] < self.header_footer_margin
                is_footer = bbox[3] > page_height - self.header_footer_margin

                text_content = ""
                font_info = []

                for line in block["lines"]:
                    for span in line["spans"]:
                        text_content += span["text"]
                        font_info.append({
                            "size": span["size"],
                            "font": span["font"],
                            "color": span["color"]
                        })

                if text_content.strip():
                    cleaned_text = text_content.strip()
                    structure = self.identify_legal_structure(cleaned_text)

                    page_elements.append({
                        "text": cleaned_text,
                        "bbox": bbox,
                        "is_header": is_header,
                        "is_footer": is_footer,
                        "font_info": font_info,
                        "page_number": layout_info["page_number"],
                        "layout": layout_info,
                        "type": "text",
                        "structure": structure
                    })

            # Add tables as page elements
            for table in tables:
                table["layout"] = layout_info
                page_elements.append(table)

            # SORT elements by vertical position (y0) then horizontal (x0) to preserve reading order
            page_elements.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
            extracted_content.extend(page_elements)

        doc.close()
        return extracted_content

if __name__ == "__main__":
    # Quick test
    processor = LegalDocProcessor()
    # Use one of the existing files for a quick check if needed,
    # but the plan has a separate verification step.
    pass
