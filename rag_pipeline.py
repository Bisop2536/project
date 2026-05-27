import os
import pickle
import numpy as np
from typing import List, Dict, Any
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

class LegalRAGPipeline:
    """
    Enhanced RAG Pipeline for legal documents using Gemini Flash 2.5.
    """
    def __init__(self, db_path_prefix: str, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_id = "gemini-2.0-flash"

        # Load metadata and model info
        pkl_path = db_path_prefix + ".pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.embedding_model_name = data["model_name"]

        print(f"Loading embedding model for retrieval: {self.embedding_model_name}")
        self.embed_model = SentenceTransformer(self.embedding_model_name)

        # Load FAISS index natively
        faiss_path = db_path_prefix + ".faiss"
        self.vector_store = FAISS.load_local(
            faiss_path,
            embeddings=lambda x: self.embed_model.encode([x], normalize_embeddings=True)[0],
            allow_dangerous_deserialization=True # Necessary for loading pickled FAISS from local
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant legal chunks.
        """
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        return retrieved_docs

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generates a grounded legal response using Gemini Flash.
        """
        context_str = "\n\n".join([
            f"Source: {c['metadata']['source']}, Page: {c['metadata'].get('page', 'N/A')}\nContent: {c['content']}"
            for c in context
        ])

        prompt = f"""
You are an expert legal assistant. Use the following extracted sections from legal documents to answer the user's query.
Your answer must be accurate, preserve legal meaning, and include specific citations (document name and page number).
If the information is not present in the context, state that you do not have enough information.

Context:
{context_str}

User Query: {query}

Legal Analysis and Answer:
"""

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                max_output_tokens=1024
            )
        )

        return response.text

    def query(self, user_query: str) -> str:
        """
        Full RAG pipeline: retrieval followed by generation.
        """
        print(f"Querying: {user_query}")
        context = self.retrieve(user_query)
        print(f"Retrieved {len(context)} relevant chunks.")
        answer = self.generate_response(user_query, context)
        return answer

if __name__ == "__main__":
    # Test stub
    pass
