import os
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from google import genai
from legal_processor import LegalDocProcessor
from legal_chunker import LegalChunker

class LegalRAGPipeline:
    """
    Robust Legal RAG Pipeline using Gemini Flash and FAISS.
    """
    def __init__(self, api_key: str = None, embed_model_name: str = "BAAI/bge-base-en-v1.5"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.embed_model = SentenceTransformer(embed_model_name)
        self.processor = LegalDocProcessor()
        self.chunker = LegalChunker()
        self.vector_store = None
        self.documents = []

    def build_index(self, pdf_folder: str, save_prefix: str = "legal_index"):
        """Extracts content from PDFs, embeds, and saves to FAISS and PKL."""
        all_chunks = []
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            content = self.processor.extract_content(os.path.join(pdf_folder, pdf_file))
            all_chunks.extend(self.chunker.chunk_content(content, pdf_file))

        print(f"Embedding {len(all_chunks)} chunks...")
        texts = [doc.page_content for doc in all_chunks]
        embeddings = self.embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        self.vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=lambda x: self.embed_model.encode([x], normalize_embeddings=True)[0],
            metadatas=[doc.metadata for doc in all_chunks]
        )
        self.vector_store.save_local(save_prefix + ".faiss")
        with open(save_prefix + ".pkl", "wb") as f:
            pickle.dump({"documents": all_chunks, "model_name": "BAAI/bge-base-en-v1.5"}, f)
        print(f"Index saved as {save_prefix}.faiss and {save_prefix}.pkl")

    def load_index(self, load_prefix: str = "legal_index"):
        """Loads the FAISS index and metadata."""
        with open(load_prefix + ".pkl", "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.vector_store = FAISS.load_local(
            load_prefix + ".faiss",
            embeddings=lambda x: self.embed_model.encode([x], normalize_embeddings=True)[0],
            allow_dangerous_deserialization=True
        )
        print("Index loaded successfully.")

    def query(self, user_query: str, top_k: int = 5) -> str:
        """Retrieves context and generates a grounded response using Gemini Flash."""
        if not self.vector_store: raise ValueError("Index not loaded.")
        if not self.api_key: raise ValueError("GOOGLE_API_KEY not set.")

        client = genai.Client(api_key=self.api_key)
        results = self.vector_store.similarity_search(user_query, k=top_k)

        context = "\n\n".join([
            f"Source: {d.metadata['source']}, Page: {d.metadata['page']}\nContent: {d.page_content}"
            for d in results
        ])

        prompt = f"""
You are a legal expert. Answer the following query based ONLY on the provided context.
Include specific citations (source name and page number).

Query: {user_query}

Context:
{context}

Answer:
"""
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text

if __name__ == "__main__":
    # Example execution:
    # pipeline = LegalRAGPipeline()
    # pipeline.build_index("./pdfs")
    # print(pipeline.query("What are the environmental protection rules?"))
    pass
