import os
from legal_processor import LegalDocProcessor
from legal_chunker import LegalChunker
from vector_storage import VectorStorage
from rag_pipeline import LegalRAGPipeline

def main():
    # 1. Setup
    pdf_folder = "." # Current directory contains the PDFs
    index_prefix = "legal_vector_db"

    # 2. Extract and Index (if not already done)
    if not os.path.exists(index_prefix + ".pkl"):
        print("Starting document extraction and indexing...")
        processor = LegalDocProcessor()
        chunker = LegalChunker()
        storage = VectorStorage()

        # In a real scenario, we'd list all PDFs
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        # For demo, let's just do a few
        # pdf_files = pdf_files[:5]

        all_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Processing {pdf_file}...")
            content = processor.extract_text_with_layout(pdf_path)
            chunks = chunker.chunk_content(content, pdf_file)
            all_chunks.extend(chunks)

        print(f"Total chunks: {len(all_chunks)}")
        storage.create_and_save_index(all_chunks, index_prefix)
    else:
        print(f"Index {index_prefix} found. Loading...")

    # 3. Query the RAG Pipeline
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        pipeline = LegalRAGPipeline(db_path_prefix=index_prefix, api_key=api_key)
        query = "What are the penalties for polluting water according to the National Environmental Act?"
        print(f"\nUser Query: {query}")
        answer = pipeline.query(query)
        print("\n--- Answer ---")
        print(answer)
    else:
        print("\nGOOGLE_API_KEY not found. Skipping RAG query demonstration.")

if __name__ == "__main__":
    main()
