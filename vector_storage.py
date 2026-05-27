import os
import pickle
import numpy as np
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

class VectorStorage:
    """
    Handles embedding generation and FAISS vector database storage.
    """
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Generates embeddings for a list of LangChain Documents.
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings

    def create_and_save_index(self, documents: List[Document], save_path_prefix: str):
        """
        Creates a FAISS index and serializes the database.
        """
        embeddings = self.embed_documents(documents)

        # We use a lambda for the embedding function as FAISS from_embeddings expects it
        faiss_index = FAISS.from_embeddings(
            text_embeddings=list(zip([doc.page_content for doc in documents], embeddings)),
            embedding=lambda x: self.embedding_model.encode([x], normalize_embeddings=True)[0],
            metadatas=[doc.metadata for doc in documents]
        )

        # Save FAISS index natively
        faiss_save_path = save_path_prefix + ".faiss"
        faiss_index.save_local(faiss_save_path)

        # Serialize documents and model info to .pkl
        pkl_save_path = save_path_prefix + ".pkl"
        data_to_save = {
            "documents": documents,
            "model_name": self.model_name
        }

        with open(pkl_save_path, "wb") as f:
            pickle.dump(data_to_save, f)

        print(f"Index saved with prefix {save_path_prefix}")
        return faiss_index

if __name__ == "__main__":
    # Test stub
    pass
