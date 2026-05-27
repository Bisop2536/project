# Legal RAG Pipeline Implementation

This repository provides a modular and robust RAG pipeline specifically designed for legal document analysis.

## Project Structure
- `legal_processor.py`: Handles high-accuracy PDF extraction, layout analysis, and reading order preservation.
- `legal_chunker.py`: Implements structure-aware chunking optimized for legal sections and clauses.
- `rag_pipeline.py`: Consolidates the RAG logic, combining FAISS-based retrieval and Gemini Flash generation.

## Prerequisites
Install the required libraries:
```bash
pip install pymupdf pandas tabulate sentence-transformers faiss-cpu google-genai langchain langchain-community
```

## Setup and Usage

### 1. Indexing Documents
To create the vector database and metadata file from your PDF documents:

```python
from rag_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()
# Specify the folder containing your legal PDF files
pipeline.build_index(pdf_folder="./path_to_pdfs", save_prefix="legal_index")
```
This will create:
- `legal_index.faiss/`: Binary FAISS index for vector search.
- `legal_index.pkl`: Metadata mapping containing original documents and chunk info.

### 2. Querying the Pipeline
Set your Google API Key and run queries:

```python
import os
from rag_pipeline import LegalRAGPipeline

os.environ["GOOGLE_API_KEY"] = "your_actual_api_key"

pipeline = LegalRAGPipeline()
pipeline.load_index(load_prefix="legal_index")

response = pipeline.query("What are the penalties for environmental violations?")
print(response)
```

## Technical Highlights
- **Layout Awareness**: Preserves natural reading order by sorting elements by coordinates.
- **Table Preservation**: Extracts tables into Markdown for superior LLM semantic understanding.
- **Semantic Chunking**: Avoids splitting critical legal clauses and utilizes word-aware overlaps.
- **Gemini Flash 2.0/2.5**: Leverages the latest Gemini models for accurate legal reasoning.
