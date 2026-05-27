from typing import List, Dict, Any
from langchain_core.documents import Document

class LegalChunker:
    """
    Advanced chunking strategy for legal documents.
    Preserves semantic and legal context by avoiding splitting clauses and sections.
    """
    def __init__(self, target_chunk_size: int = 1000, overlap: int = 100):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_content(self, extracted_content: List[Dict[str, Any]], doc_name: str) -> List[Document]:
        """
        Chunks the extracted content while respecting legal structure.
        """
        documents = []
        current_chunk_text = ""
        current_metadata = {
            "source": doc_name,
            "page_numbers": set(),
            "sections": set(),
            "layout_types": set(),
            "orientations": set()
        }

        for block in extracted_content:
            block_text = block["text"]
            block_type = block["type"]
            block_page = block["page_number"]
            block_layout = block["layout"]

            # Start a new chunk if current one is getting large or if it's a new section header
            is_section_header = block.get("structure", {}).get("is_section_header", False)

            if (len(current_chunk_text) + len(block_text) > self.target_chunk_size and current_chunk_text) or is_section_header:
                if current_chunk_text:
                    # Finalize current chunk
                    doc_metadata = self._finalize_metadata(current_metadata)
                    documents.append(Document(
                        page_content=current_chunk_text.strip(),
                        metadata=doc_metadata
                    ))

                    # Start new chunk with overlap if not a section header
                    if not is_section_header:
                        # Robust overlap: take last 'overlap' characters and adjust to nearest word
                        raw_overlap = current_chunk_text[-self.overlap:] if len(current_chunk_text) > self.overlap else current_chunk_text
                        # Find first whitespace to avoid cutting a word in half
                        first_space = raw_overlap.find(" ")
                        if first_space != -1 and first_space < len(raw_overlap) // 2:
                            overlap_text = raw_overlap[first_space:].strip()
                        else:
                            overlap_text = raw_overlap.strip()
                        current_chunk_text = overlap_text + " "
                    else:
                        current_chunk_text = ""

                    # Reset metadata
                    current_metadata = {
                        "source": doc_name,
                        "page_numbers": set(),
                        "sections": set(),
                        "layout_types": set(),
                        "orientations": set()
                    }

            current_chunk_text += block_text + " "
            current_metadata["page_numbers"].add(block_page)
            current_metadata["layout_types"].add(block_type)
            current_metadata["orientations"].add("landscape" if block_layout["is_landscape"] else "portrait")

            if is_section_header:
                current_metadata["sections"].add(block["structure"]["section_number"])

        # Add last chunk
        if current_chunk_text:
            doc_metadata = self._finalize_metadata(current_metadata)
            documents.append(Document(
                page_content=current_chunk_text.strip(),
                metadata=doc_metadata
            ))

        return documents

    def _finalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts sets in metadata to serializable formats.
        """
        final = metadata.copy()
        final["page_numbers"] = sorted(list(metadata["page_numbers"]))
        final["sections"] = sorted(list(metadata["sections"]))
        final["layout_types"] = sorted(list(metadata["layout_types"]))
        final["orientations"] = sorted(list(metadata["orientations"]))
        # Primary page for easier reference
        final["page"] = final["page_numbers"][0] if final["page_numbers"] else 0
        return final
