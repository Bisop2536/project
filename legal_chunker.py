from typing import List, Dict, Any
from langchain_core.documents import Document

class LegalChunker:
    """
    Structure-aware chunking strategy for legal documents.
    Preserves semantic context by avoiding splitting clauses and sections.
    """
    def __init__(self, target_chunk_size: int = 1000, overlap: int = 150):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_content(self, extracted_content: List[Dict[str, Any]], doc_name: str) -> List[Document]:
        documents = []
        current_chunk_text = ""
        current_metadata = {"source": doc_name, "page_numbers": set(), "sections": set()}

        for block in extracted_content:
            is_section = block.get("structure", {}).get("is_section_header", False)
            # Start new chunk if current is large or we hit a section header
            if (len(current_chunk_text) + len(block["text"]) > self.target_chunk_size and current_chunk_text) or is_section:
                if current_chunk_text:
                    documents.append(Document(page_content=current_chunk_text.strip(),
                                             metadata=self._finalize_metadata(current_metadata)))
                    # Start new chunk with word-aware overlap if not a section header
                    if not is_section:
                        raw_overlap = current_chunk_text[-self.overlap:]
                        split_idx = raw_overlap.find(" ")
                        current_chunk_text = (raw_overlap[split_idx:].strip() if split_idx != -1 else raw_overlap) + " "
                    else:
                        current_chunk_text = ""
                    current_metadata = {"source": doc_name, "page_numbers": set(), "sections": set()}

            current_chunk_text += block["text"] + " "
            current_metadata["page_numbers"].add(block["page_number"])
            if is_section:
                current_metadata["sections"].add(block["structure"]["section_number"])

        if current_chunk_text:
            documents.append(Document(page_content=current_chunk_text.strip(),
                                     metadata=self._finalize_metadata(current_metadata)))
        return documents

    def _finalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        final = metadata.copy()
        final["page_numbers"] = sorted(list(metadata["page_numbers"]))
        final["sections"] = sorted(list(metadata["sections"]))
        final["page"] = final["page_numbers"][0] if final["page_numbers"] else 0
        return final
