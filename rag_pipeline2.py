import os
import fitz
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import logging
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import functools

_embedding_model_cache = None
def get_cached_embedding_model():
    global _embedding_model_cache
    if _embedding_model_cache is None:
        print("🔄 [CACHE MISS] Loading SentenceTransformer Embedding Model into memory...")
        _embedding_model_cache = SentenceTransformer("BAAI/bge-base-en-v1.5")
        _embedding_model_cache.eval()
        print("✅ [CACHE SET] Embedding Model loaded successfully!")
    return _embedding_model_cache

_cross_encoder_cache = None
def get_cached_cross_encoder():
    global _cross_encoder_cache
    if _cross_encoder_cache is None:
        print("🔄 [CACHE MISS] Loading CrossEncoder Model into memory...")
        _cross_encoder_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ [CACHE SET] CrossEncoder loaded successfully!")
    return _cross_encoder_cache

_retrievers_cache = {}
def get_cached_retrievers(index_files_tuple, k=30):
    cache_key = (index_files_tuple, k)
    if cache_key in _retrievers_cache:
        return _retrievers_cache[cache_key]
    print(f"🔄 [CACHE MISS] Loading and stacking FAISS and BM25 databases from {len(index_files_tuple)} files...")
    all_documents = []
    all_embeddings = []
    loaded_files = []
    
    for index_file in index_files_tuple:
        if os.path.exists(index_file):
            try:
                with open(index_file, "rb") as f:
                    data = pickle.load(f)
                all_documents.extend(data["documents"])
                all_embeddings.append(data["embeddings"])
                loaded_files.append(index_file)
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to load index file '{index_file}': {e}")
                
    if not loaded_files:
        return None, None, None, None
        
    documents = all_documents
    if len(all_embeddings) > 1:
        embeddings = np.vstack(all_embeddings)
    else:
        embeddings = all_embeddings[0]
        
    texts = [doc.page_content for doc in documents]
    emb_model = get_cached_embedding_model()
    
    faiss_index = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        embedding=lambda x: emb_model.encode(x, normalize_embeddings=True),
        metadatas=[doc.metadata for doc in documents]
    )
    faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": k})
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    print(f"✅ [CACHE SET] Vector databases stacked and loaded into memory! (Total chunks: {len(documents)})")
    result = (documents, embeddings, faiss_retriever, bm25_retriever)
    _retrievers_cache[cache_key] = result
    return result

# Import agentic tools
from gemini_tools import ToolExecutor, get_tool_schemas_for_gemini, format_tool_result_for_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Very simple semantic-ish chunker:
    1) First split into small fixed chunks.
    2) Then merge adjacent chunks whose embeddings are very similar.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        base_chunk_size: int = 512,
        base_overlap: int = 50,
        sim_threshold: float = 0.85,
    ):
        from langchain_text_splitters import CharacterTextSplitter

        self.embedding_model = embedding_model
        self.base_splitter = CharacterTextSplitter(
            chunk_size=base_chunk_size,
            chunk_overlap=base_overlap,
            separator="\n\n",
        )
        self.sim_threshold = sim_threshold

    def split_text(self, text: str) -> list[str]:
        # 1) base split
        mini_chunks = self.base_splitter.split_text(text)
        if len(mini_chunks) <= 1:
            return mini_chunks

        # 2) embed all mini-chunks
        embs = self.embedding_model.encode(
            mini_chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )

        # 3) merge adjacent chunks with high cosine similarity
        merged_chunks: list[str] = []
        current = mini_chunks[0]
        current_emb = embs[0]

        for i in range(1, len(mini_chunks)):
            # Round similarity to 6 decimal places to prevent floating-point drift across CPUs
            sim = round(float(np.dot(current_emb, embs[i])), 6)
            if sim >= self.sim_threshold:
                # same topic: merge
                current = current + " " + mini_chunks[i]
                # update embedding as average of both (approx)
                current_emb = (current_emb + embs[i]) / 2.0
            else:
                merged_chunks.append(current)
                current = mini_chunks[i]
                current_emb = embs[i]

        merged_chunks.append(current)
        return merged_chunks

class EnsembleRetriever(BaseRetriever):
    """Ensemble retriever that combines multiple retrievers."""
    
    retrievers: List[BaseRetriever]
    weights: List[float]
    
    def __init__(self, retrievers: List[BaseRetriever], weights: List[float] = None):
        # Calculate weights if not provided
        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)
        
        # Call parent __init__ with the fields as keyword arguments
        super().__init__(retrievers=retrievers, weights=weights)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from all retrievers."""
        all_docs = []
        doc_scores = {}
        
        for retriever, weight in zip(self.retrievers, self.weights):
            # FIXED: Try both invoke() and get_relevant_documents() for compatibility
            try:
                docs = retriever.invoke(query)  # New LangChain method
            except AttributeError:
                try:
                    docs = retriever.get_relevant_documents(query)  # Old method
                except AttributeError:
                    logger.warning(f"Retriever {type(retriever)} has no compatible method")
                    continue
            
            for i, doc in enumerate(docs):
                doc_id = doc.page_content
                score = weight * (1.0 / (i + 1))  # Reciprocal rank fusion
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += score
                else:
                    doc_scores[doc_id] = {'doc': doc, 'score': score}
        
        # Sort by score and return documents
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version - for now just call sync version."""
        return self._get_relevant_documents(query)
    
    def invoke(self, query: str) -> List[Document]:
        """Public invoke method for new LangChain compatibility."""
        return self._get_relevant_documents(query)

class RelevanceChecker:
    """
    RelevanceChecker: rerank, threshold, optional contextual compression.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        cross_encoder_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        threshold: float = 0.70,
        min_docs: int = 2,
        max_docs: int = 6,
        enable_compression: bool = True,
        compression_top_sentences: int = 3,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        fallback_to_cosine: bool = True
    ):
        self.embedding_model = embedding_model
        self.cross_encoder = None
        self.cross_encoder_name = cross_encoder_name

        if cross_encoder_name:
            try:
                self.cross_encoder = get_cached_cross_encoder()
                logger.info(f"Loaded CrossEncoder from cache: {cross_encoder_name}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder '{cross_encoder_name}': {e}")
                self.cross_encoder = None

        self.threshold = threshold
        self.min_docs = min_docs
        self.max_docs = max_docs
        self.enable_compression = enable_compression
        self.compression_top_sentences = compression_top_sentences
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.fallback_to_cosine = fallback_to_cosine

    # -------------------------
    # Public API
    # -------------------------
    def filter_documents(
        self,
        question: str,
        docs: List,
        doc_embeddings: Optional[np.ndarray] = None
    ) -> List[Tuple[object, float]]:

        logger.info(f"Filtering {len(docs)} retrieved chunks for question: {question}")

        if not docs:
            logger.warning("No documents retrieved.")
            return []

        # 1) Scoring
        if self.cross_encoder is not None:
            scored = self._score_with_crossencoder(question, docs)
        else:
            scored = self._score_with_cosine(question, docs, doc_embeddings)

        # Log all scored chunks
        for i, (doc, score) in enumerate(scored):
            logger.info(f"[Score] Rank={i+1} Score={score:.4f} Content={doc.page_content[:150]}...")

        # 2) Thresholding
        filtered = [(d, s) for d, s in scored if s >= self.threshold]
        logger.info(f"Chunks above threshold {self.threshold}: {len(filtered)}")

        if len(filtered) < self.min_docs:
            logger.info(f"Below min_docs={self.min_docs}, selecting top {self.min_docs} anyway.")
            filtered = scored[: self.min_docs]

        filtered = filtered[: self.max_docs]

        # Log filtered results
        for doc, score in filtered:
            logger.info(f"[Selected] Score={score:.4f} Content={doc.page_content[:150]}...")

        # 3) Compression
        if self.enable_compression:
            compressed = []
            for doc, score in filtered:
                logger.info(f"Compressing chunk (score={score:.4f}): {doc.page_content[:150]}...")
                compressed_doc = self._compress_document(question, doc, top_k=self.compression_top_sentences)
                logger.info(f"[Compressed Result] {compressed_doc.page_content[:200]}...")
                compressed.append((compressed_doc, score))
            filtered = compressed

        return filtered

    # -------------------------
    # Internal scoring helpers
    # -------------------------
    def _score_with_crossencoder(self, question: str, docs: List) -> List[Tuple[object, float]]:
        # Format as tuples (query, passage)
        cross_input = [(question, doc.page_content) for doc in docs]
        try:
            scores = self.cross_encoder.predict(cross_input, batch_size=self.batch_size)
            logger.info("Cross-encoder scoring successful.")
        except Exception as e:
            logger.warning(f"Cross-encoder failed: {e}, falling back to cosine.")
            return self._score_with_cosine(question, docs)

        scores = self._minmax_normalize(np.array(scores))
        return sorted(list(zip(docs, scores.tolist())), key=lambda x: x[1], reverse=True)


    def _score_with_cosine(self, question: str, docs: List, doc_embeddings: Optional[np.ndarray] = None):
        logger.info("Using cosine similarity scoring...")

        q_emb = self.embedding_model.encode([question], show_progress_bar=False, convert_to_numpy=True)
        if self.normalize_embeddings:
            q_emb = self._l2_normalize(q_emb)

        if doc_embeddings is None:
            texts = [d.page_content for d in docs]
            doc_embeddings = self.embedding_model.encode(texts, batch_size=self.batch_size,
                                                         show_progress_bar=False, convert_to_numpy=True)
        if self.normalize_embeddings:
            doc_embeddings = self._l2_normalize(doc_embeddings)

        sims = np.dot(doc_embeddings, q_emb.T).reshape(-1)
        sims = (sims + 1.0) / 2.0  # normalize to 0–1
        # Round to 6 decimal places for deterministic thresholding across CPUs
        sims = np.round(sims, 6)

        return sorted(list(zip(docs, sims.tolist())), key=lambda x: x[1], reverse=True)

    # -------------------------
    # Contextual compression
    # -------------------------
    def _compress_document(self, question: str, doc, top_k: int = 3):
        text = doc.page_content
        sentences = self._split_sentences(text)

        logger.info(f"Splitting into {len(sentences)} sentences for compression.")

        if not sentences:
            return doc

        q_emb = self.embedding_model.encode([question], show_progress_bar=False, convert_to_numpy=True)
        sent_embs = self.embedding_model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)

        if self.normalize_embeddings:
            q_emb = self._l2_normalize(q_emb)
            sent_embs = self._l2_normalize(sent_embs)

        sims = np.dot(sent_embs, q_emb.T).reshape(-1)

        # Log top sentences
        idx_scores = list(enumerate(sims))
        idx_scores_sorted = sorted(idx_scores, key=lambda x: x[1], reverse=True)
        for i, (idx, score) in enumerate(idx_scores_sorted[:top_k]):
            logger.info(f"[Compression Sentence #{i+1}] Score={score:.4f} Sentence={sentences[idx][:200]}")

        top_idx = [i for i, _ in idx_scores_sorted[:top_k]]
        top_idx.sort()

        compressed_text = " ".join([sentences[i] for i in top_idx]).strip()

        compressed_doc = type(doc)(
            page_content=compressed_text,
            metadata={**getattr(doc, "metadata", {})}
        )
        return compressed_doc

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _minmax_normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-12)

    @staticmethod
    def _l2_normalize(x):
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom[denom == 0] = 1e-12
        return x / denom

    @staticmethod
    def _split_sentences(text: str):
        parts = re.split(r'(?<=[\.\?\!])\s+', text)
        return [p.strip() for p in parts if p.strip()]
class ConversationManager:
    """Manages conversation history with model-aware token limits"""
    
    MODEL_LIMITS = {
        "gemini": 32768,      # Gemini 1.5 Flash (Gemma 4B)
    }
    
    # 
    #     Args:
    #         llm_type: "gemini" for Gemma 4B via Gemini API
    #         reserve_tokens: Tokens to reserve for system prompt + retrieved docs + response
    #     
    def __init__(self, llm_type: str = "gemini", reserve_tokens: int = 8000):
        if llm_type not in self.MODEL_LIMITS:
            logger.warning(f"Unknown llm_type: {llm_type}. Defaulting to 'gemini'.")
            llm_type = "gemini"
        
        self.llm_type = llm_type
        self.max_context = self.MODEL_LIMITS[llm_type]
        self.reserve_tokens = reserve_tokens
        
        # Available tokens for conversation history
        self.available_for_history = self.max_context - reserve_tokens
        
        self.history: List[Dict[str, str]] = []
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load tiktoken: {e}. Using character-based fallback.")
            self.tokenizer = None
        
        
        logger.info(f"ConversationManager initialized: {self.llm_type}, "
                   f"max={self.max_context}, available_for_history={self.available_for_history}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list"""
        total = 0
        for msg in messages:
            # Count content tokens
            total += self.count_tokens(msg["content"])
            # Add overhead for message formatting (~4 tokens per message)
            total += 4
        return total
    
    def add_exchange(self, user_message: str, assistant_message: str):
        """Add a Q&A pair to history with automatic truncation"""
        # Add new messages
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        
        # Truncate if needed
        self._truncate_to_fit()
    
    def _truncate_to_fit(self):
        """Remove oldest messages until history fits in available token budget"""
        current_tokens = self.count_messages_tokens(self.history)
        
        # Keep removing oldest Q&A pairs until we fit
        while current_tokens > self.available_for_history and len(self.history) > 2:
            # Remove oldest Q&A pair (first 2 messages)
            removed = self.history[:2]
            self.history = self.history[2:]
            
            removed_tokens = self.count_messages_tokens(removed)
            current_tokens -= removed_tokens
            
            logger.info(f"Truncated conversation: removed {removed_tokens} tokens, "
                       f"remaining={current_tokens}/{self.available_for_history}")
        
        # Log if we're getting close to limit
        if current_tokens > self.available_for_history * 0.8:
            pairs = len(self.history) // 2
            logger.warning(f"Conversation history at 80% capacity: {current_tokens} tokens, {pairs} pairs")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.history.copy()

    def set_history(self, messages: List[Dict[str, str]]):
        """Set history from external message list, ensuring token limits are respected"""
        self.history = []
        for msg in messages:
            if msg.get("role") in ["user", "assistant"]:
                self.history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        self._truncate_to_fit()
    
    def get_history_tokens(self) -> int:
        """Get current token count of history"""
        return self.count_messages_tokens(self.history)
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict:
        """Get conversation statistics"""
        pairs = len(self.history) // 2
        tokens = self.count_messages_tokens(self.history)
        
        return {
            "total_exchanges": pairs,
            "history_tokens": tokens,
            "available_tokens": self.available_for_history,
            "utilization_percent": round((tokens / self.available_for_history) * 100, 1),
            "model": self.llm_type,
            "max_context": self.max_context
        }
#sentence-transformers/all-mpnet-base-v2
class PDFExtractor:
    """Handles PDF extraction with layout preservation"""
    
    def __init__(self):
        self.header_footer_margin = 50
        self.min_text_length = 50
        self.heading_font_threshold = 13
    
    def extract_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract content from PDF with structure preservation
        Returns: List of content blocks with metadata
        """
        try:
            content_blocks = self._extract_with_layout(pdf_path)
            
            if not content_blocks:
                logger.warning(f"Layout extraction failed, using fallback for {pdf_path}")
                content_blocks = self._fallback_extraction(pdf_path)
            
            # Merge small adjacent blocks
            content_blocks = self._merge_blocks(content_blocks)
            
            return content_blocks
        
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            return []
    
    def _extract_with_layout(self, pdf_path: str) -> List[Dict]:
        """Extract text with layout and structure preservation"""
        doc = fitz.open(pdf_path)
        all_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_height = page.rect.height
            
            # Get text blocks with layout info
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                bbox = block["bbox"]
                
                # Skip headers and footers
                if (bbox[1] < self.header_footer_margin or 
                    bbox[3] > page_height - self.header_footer_margin):
                    continue
                
                # Extract text and font information
                text_lines = []
                font_sizes = []
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_sizes.append(span["size"])
                    
                    if line_text.strip():
                        text_lines.append(line_text.strip())
                
                if not text_lines:
                    continue
                
                text = " ".join(text_lines)
                
                if len(text.strip()) < self.min_text_length:
                    continue
                
                # Detect headings by font size
                avg_font_size = np.mean(font_sizes) if font_sizes else 11
                content_type = "heading" if avg_font_size > self.heading_font_threshold else "paragraph"
                
                all_content.append({
                    "text": text,
                    "page": page_num + 1,
                    "type": content_type,
                    "bbox": bbox
                })
            
            # Extract tables separately
            tables = self._extract_tables(page, page_num + 1)
            all_content.extend(tables)
        
        doc.close()
        return all_content
    
    def _extract_tables(self, page, page_num: int) -> List[Dict]:
        """Extract tables from page using PyMuPDF"""
        tables = []
        
        try:
            tabs = page.find_tables()
            
            for i, table in enumerate(tabs):
                df = table.to_pandas()
                
                if df.empty:
                    continue
                
                table_text = f"Table {i+1}:\n{df.to_string(index=False)}"
                
                tables.append({
                    "text": table_text,
                    "page": page_num,
                    "type": "table",
                    "bbox": table.bbox
                })
        
        except Exception as e:
            logger.debug(f"Table extraction failed on page {page_num}: {e}")
        
        return tables
    
    def _fallback_extraction(self, pdf_path: str) -> List[Dict]:
        """Simple fallback if advanced extraction fails"""
        doc = fitz.open(pdf_path)
        content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            if text.strip():
                content.append({
                    "text": text.strip(),
                    "page": page_num + 1,
                    "type": "text"
                })
        
        doc.close()
        return content
    
    def _merge_blocks(self, content_blocks: List[Dict]) -> List[Dict]:
        """Merge small adjacent blocks on same page"""
        if not content_blocks:
            return []
        
        merged = []
        current_block = None
        
        for block in content_blocks:
            # Always keep tables separate
            if block["type"] == "table":
                if current_block:
                    merged.append(current_block)
                    current_block = None
                merged.append(block)
                continue
            
            if current_block is None:
                current_block = block.copy()
                continue
            
            # Merge if same page and combined text not too long
            if (block["page"] == current_block["page"] and 
                len(current_block["text"]) + len(block["text"]) < 800):
                current_block["text"] += " " + block["text"]
            else:
                merged.append(current_block)
                current_block = block.copy()
        
        if current_block:
            merged.append(current_block)
        
        return merged
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d{1,3}\s*$', '', text)
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\"\'\%\$\#\@\!\?\/\&\+\=\*]', '', text)
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        return text.strip()


class RAGPipeline2:
    """Main RAG pipeline for PDF question answering with Gemini API (Gemma 12B)"""
    def __init__(
        self,
        pdf_folder: str,
        index_file: str,
        model_params: dict,
        reserve_tokens: int = 8000,
        gemini_rotator=None,
    ):
        # Paths
        self.pdf_folder = pdf_folder
        self.index_file = index_file
        
        # Configure Gemini API with first key or rotator
        self.gemini_rotator = gemini_rotator
        if self.gemini_rotator:
            idx, key = self.gemini_rotator.get_next_key()
            genai.configure(api_key=key)
            self.current_key_idx = idx
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            print(f"🚀 [Marsh Thinking] Initialized with Google Gemini API Key #{idx + 1} ({masked_key})")
        elif "google_api_key" in model_params:
            key = model_params["google_api_key"]
            genai.configure(api_key=key)
            self.current_key_idx = None
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            print(f"🚀 [Marsh Thinking] Initialized with static Google Gemini API Key ({masked_key})")
        else:
            raise ValueError("Either model_params['google_api_key'] or gemini_rotator is required")
        
        # Initialize Gemini model with Gemma 3 12B
        # Using models/gemma-3-12b-it which will use an enhanced manual ReAct loop
        self.model_name = model_params.get("model_name", "models/gemma-3-12b-it")
        
        # Create generative model - NOT using tools parameter for Gemma manual loop
        # Enforce determinism with temperature 0.0
        # Added safety settings to prevent false positives when analyzing environmental policy/penalties
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        self.llm_client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                # ── DETERMINISM LOCK ──────────────────────────────────────────
                # temperature=0.0 + top_k=1 + top_p=1.0 → pure greedy decoding.
                # The model ALWAYS picks the single highest-probability token.
                # This guarantees byte-for-byte identical answers across any
                # device, browser, OS, or session for the same conversation state.
                "temperature": 0.0,
                "top_p": 1.0,           # disable nucleus sampling
                "top_k": 1,             # greedy: always pick #1 token
                "max_output_tokens": 4500,  # allow detailed, comprehensive answers
            },
            safety_settings=safety_settings
        )
        
        self.last_retrieved_docs = []
        
        logger.info(f"Initialized Deterministic Gemini model: {self.model_name}")
        
        # Initialize conversation manager for Gemini
        self.conversation_manager = ConversationManager(
            llm_type="gemini",
            reserve_tokens=reserve_tokens
        )
        
        # Initialize tool executor (will be set after retrievers are built)
        self.tool_executor = None
        
        # Models
        self.embedding_model = get_cached_embedding_model()
        # create relevance checker
        self.relevance_checker = RelevanceChecker(
            embedding_model=self.embedding_model,
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            # Lowered threshold 0.70 → 0.60: don't drop borderline-relevant chunks
            # that may contain exact policy numbers, penalties, or dates.
            threshold=0.60,
            min_docs=3,
            max_docs=12,
            # LOCKED False: sentence-level compression uses cosine similarity
            # comparisons that can reorder in floating-point edge cases → non-deterministic.
            enable_compression=False,
            compression_top_sentences=3
        )
        
        self.pdf_extractor = PDFExtractor()
        
        # Text splitter
        # self.text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=700,
        #     chunk_overlap=150,
        #     length_function=len,
        #     separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        #     is_separator_regex=False,
        # )

        #Simple sliding window chunker.
        # from langchain_text_splitters import CharacterTextSplitter

        # self.text_splitter = CharacterTextSplitter(
        #     chunk_size=512,
        #     chunk_overlap=50,
        #     separator="\n\n",   
        # )

        #Simple fixed length chunker.
        # from langchain_text_splitters import CharacterTextSplitter

        # self.text_splitter = CharacterTextSplitter(
        #     chunk_size=512,
        #     chunk_overlap=0,
        #     separator="\n\n",   
        # )

        #Semantic-ish chunker.
        self.text_splitter = SemanticChunker(
            embedding_model=self.embedding_model,
            base_chunk_size=512,
            base_overlap=50,
            sim_threshold=0.85,
        )
        """The Semantica actually worked better than the sliding window chunker."""
        """Fixed length is shape. But it migth not be good to pick up on context."""
        """Sliding window chunker actually made it better. I'll have to tweak more with the chunk_size and chunk_overlap."""
        """When I checked the chunks found in recursive splitter it did have the required texts, but the recursive splitter
           seems to split them at random points leading to a losss of context."""
        
        # Storage
        self.documents = []
        self.embeddings = None
        self.faiss_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
    
    def build_index(self, progress_callback=None, status_callback=None):
        """Build index from PDFs in folder"""
        # Deterministic: sort file names to ensure identical indexing on all machines
        pdf_files = sorted([f for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")])
        
        if not pdf_files:
            raise ValueError("No PDF files found in folder")
        
        all_documents = []
        
        for i, pdf_file in enumerate(pdf_files):
            if status_callback:
                status_callback(f"Processing: {pdf_file} ({i+1}/{len(pdf_files)})")
            
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            
            try:
                # Extract content blocks
                content_blocks = self.pdf_extractor.extract_pdf(pdf_path)
                
                # Create smart chunks
                documents = self._create_chunks(content_blocks, pdf_file)
                all_documents.extend(documents)
                
                logger.info(f"Extracted {len(documents)} chunks from {pdf_file}")
            
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue
            
            if progress_callback:
                progress_callback((i + 1) / len(pdf_files))
        
        if not all_documents:
            raise ValueError("No content extracted from PDFs")
        
        if status_callback:
            status_callback(f"Encoding {len(all_documents)} chunks...")
        
        # Encode documents
        texts = [doc.page_content for doc in all_documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Build retrievers
        self._build_retrievers(all_documents, texts, embeddings)
        
        # Save index
        self.documents = all_documents
        self.embeddings = embeddings
        self._save_index()
        
        if status_callback:
            status_callback(f"✅ Indexed {len(all_documents)} chunks from {len(pdf_files)} PDFs")
        
        return len(all_documents)
    
    def _create_chunks(self, content_blocks: List[Dict], pdf_name: str) -> List[Document]:
        """Create smart chunks from content blocks"""
        documents = []
        
        for block in content_blocks:
            text = self.pdf_extractor.clean_text(block["text"])
            
            if len(text) < 50:
                continue
            
            # Keep tables and short content as single chunks
            if block["type"] == "table" or len(text) < 600:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_name,
                        "page": block["page"],
                        "type": block["type"]
                    }
                ))
            else:
                # Split long content
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_name,
                            "page": block["page"],
                            "type": block["type"],
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    ))
        
        return documents
    
    def _build_retrievers(self, documents: List[Document], texts: List[str], embeddings: np.ndarray):
        """Build FAISS and BM25 retrievers"""
        # FAISS retriever
        faiss_index = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=lambda x: self.embedding_model.encode(x, normalize_embeddings=True),
            metadatas=[doc.metadata for doc in documents]
        )
        self.faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": 30})
        
        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 30
        
        # Hybrid retriever
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.faiss_retriever, self.bm25_retriever],
            weights=[0.85, 0.15]
        )
        
        # Initialize tool executor
        from gemini_tools import ToolExecutor
        self.tool_executor = ToolExecutor(self)
        logger.info("ToolExecutor initialized for RAGPipeline2")
    
    def _save_index(self):
        """Save index to disk"""
        with open(self.index_file, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "model": "sentence-transformers/all-mpnet-base-v2"
            }, f)
    
    def load_index(self):
        """Load existing index from disk (supports single file or list of files) using cache"""
        index_files = self.index_file if isinstance(self.index_file, list) else [self.index_file]
        
        docs_and_rets = get_cached_retrievers(tuple(index_files), k=30)
        
        if docs_and_rets[0] is None:
            logger.error("No index files were loaded successfully")
            return False
            
        self.documents, self.embeddings, self.faiss_retriever, self.bm25_retriever = docs_and_rets
        
        # Hybrid retriever
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.faiss_retriever, self.bm25_retriever],
            weights=[0.85, 0.15]
        )
        
        # Initialize tool executor
        from gemini_tools import ToolExecutor
        self.tool_executor = ToolExecutor(self)
        logger.info("ToolExecutor initialized directly from cache for RAGPipeline2")
        
        logger.info(f"Loaded merged index with {len(self.documents)} chunks from {len(index_files)} files (Cached)")
        return True
    
    def _normalize_query(self, question: str) -> str:
        """
        Canonical query preprocessing: strips filler words so that semantically
        identical questions from different typings are normalized before retrieval.
        Only used for retrieval; the original question is still fed to the LLM.
        """
        q = question.strip()
        # Remove common filler prefixes that add no retrieval signal
        filler_patterns = [
            r'^(?:please|kindly)\s+',
            r'^(?:can you|could you|would you|will you)\s+',
            r'^(?:tell me|explain|describe|show me|give me|provide)\s+(?:about\s+|me\s+)?',
            r'^(?:what is|what are|what were|what was)\s+',
            r'^(?:how does|how do|how can|how to)\s+',
            r'^(?:i want to know|i need to know|i would like to know)\s+',
        ]
        q_lower = q.lower()
        for pattern in filler_patterns:
            q_lower = re.sub(pattern, '', q_lower, flags=re.IGNORECASE).strip()
        # Return the cleaned lowercase version for retrieval queries
        return q_lower if q_lower else q.lower()

    def _expand_queries(self, question: str) -> List[str]:
        """
        Generate multiple canonical phrasings of the same question.
        This guarantees that semantically identical questions asked in different
        ways from different users always surface the same chunks via RRF fusion.
        """
        base = question.strip()
        normalized = self._normalize_query(base)

        # Template expansions that cover common rephrasing patterns
        expansions = [
            normalized,                                    # normalized base
            base,                                          # original verbatim
            f"definition of {normalized}",                 # definitional framing
            f"{normalized} policy regulations rules",      # policy framing
            f"{normalized} penalties fines violations",    # enforcement framing
        ]
        # Deduplicate while preserving order
        seen, unique = set(), []
        for q in expansions:
            q_stripped = q.strip().lower()
            if q_stripped and q_stripped not in seen:
                seen.add(q_stripped)
                unique.append(q.strip())
        return unique[:4]  # max 4 to avoid over-retrieval latency

    def _rrf_fuse(self, ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion: combine multiple ranked lists of documents.
        Each document's fused score = sum(1 / (k + rank_i)) across all lists.
        Deterministic: RRF score is purely arithmetic, no randomness.
        """
        scores: Dict[str, Dict] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                key = doc.page_content  # use content as identity
                if key not in scores:
                    scores[key] = {'doc': doc, 'score': 0.0}
                # Use 6 decimal places for RRF arithmetic stability
                scores[key]['score'] = round(scores[key]['score'] + 1.0 / (k + rank), 6)
        sorted_items = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_items]

    def _expand_and_retrieve(self, question: str, top_k: int = 15) -> List[Document]:
        """
        Multi-query retrieval with RRF fusion.
        Runs retrieval for each query expansion and merges results so that
        the same document surfaces regardless of how the question was phrased.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        queries = self._expand_queries(question)
        logger.info(f"Multi-query retrieval: {len(queries)} query variants (Parallel)")

        all_ranked_lists = []
        unique_queries = list(dict.fromkeys([q.lower() for q in queries])) # dedupe
        
        def _get_docs(q):
            try:
                msg = f"  Query variant '{q[:60]}'"
                docs = self.hybrid_retriever.invoke(q)
                if docs:
                    logger.info(f"{msg} returned {len(docs)} docs")
                    return docs
            except Exception as e:
                logger.warning(f"Retrieval failed for query variant '{q}': {e}")
            return []

        # Run retrievals in parallel
        with ThreadPoolExecutor(max_workers=len(unique_queries)) as executor:
            results = list(executor.map(_get_docs, unique_queries))
        
        all_ranked_lists = [r for r in results if r]

        if not all_ranked_lists:
            return []

        # Fuse all ranked lists via RRF
        fused = self._rrf_fuse(all_ranked_lists)
        logger.info(f"RRF fusion produced {len(fused)} unique docs from {len(all_ranked_lists)} query variants")
        return fused[:top_k * 3]  # return a generous pool for re-ranking

    def query(self, question: str, top_k: int = 15, **kwargs) -> str:
        """Processes a query and returns the full answer string"""
        if self.hybrid_retriever is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")

        # Ensure top_k is int
        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            top_k = 15

        # ── STEP 1: Multi-query retrieval with RRF fusion ─────────────────────
        retrieved_docs = self._expand_and_retrieve(question, top_k=top_k)

        # ── STEP 2: Re-rank fused pool with CrossEncoder ──────────────────────
        top_docs = retrieved_docs[:top_k * 2] if retrieved_docs else []
        filtered = self.relevance_checker.filter_documents(question, top_docs)
        filtered_docs = [d for d, s in filtered]

        # Capture filtered_docs for UI references
        self.last_retrieved_docs = filtered_docs

        # ── STEP 3: Generate answer via deep research ReAct loop ──────────────
        # Detect informative intent keywords in the question
        informative_keywords = [
            "informative answer", "informative", "explain in detail", "in detail",
            "comprehensive", "elaborate", "describe", "give details", "detailed"
        ]
        q_lower = question.lower()
        is_informative = any(kw in q_lower for kw in informative_keywords)
        
        # ── FALLBACK: If retrieval is sparse, use Fast Track (Direct) loop to save time ──
        if is_informative and len(filtered_docs) < 3:
            logger.info(f"Sparse retrieval ({len(filtered_docs)} docs). Falling back to Fast Track for accuracy.")
            is_informative = False

        if is_informative:
            augmented_question = (
                f"[RESEARCH DIRECTIVE: Provide a comprehensive, informative answer of 200-300 words. "
                f"Use structured headings, bullets with explanations, tables where relevant, and cite all sources.] "
                f"{question}"
            )
        else:
            augmented_question = question

        full_answer = self._generate_answer(augmented_question, filtered_docs, is_informative=is_informative)

        # Store this exchange in history once complete
        self.conversation_manager.add_exchange(question, full_answer.strip())

        # Log conversation stats
        stats = self.conversation_manager.get_stats()
        logger.info(f"Query complete. Total exchanges: {stats['total_exchanges']}")
        return full_answer


    def set_history(self, messages: List[Dict[str, str]]):
        """Synchronize conversation history"""
        self.conversation_manager.set_history(messages)

    def _generate_answer(self, question: str, context_docs: List[Document], is_informative: bool = False) -> str:
        """Generates answer using the Enhanced Manual ReAct Loop — Deep Research Mode (Non-streaming)"""

        FEW_SHOT_EXAMPLE = """
Example Reasoning Cycle:
Question: What are the specific protection rules for the Muturajawela wetland?
Thought: I need to check the document list first to find the right document, then search within it.
Action: get_document_list({})
Observation: [get_document_list]: Available documents:
- National Wetland Policy: 156 chunks, 24 pages
- Muturajawela Management Plan: 88 chunks, 12 pages
Reflection: Confidence=4/10. I found the relevant doc. Searching now for protection rules within it.
Action: search_specific_document({"document_name": "Muturajawela Management Plan", "query": "protection rules conservation regulations zones"})
Observation: [search_specific_document]: SOURCE 1: Muturajawela Management Plan (Page 4) | Relevance: HIGH
Content: Section 2.1: Zone A is strictly protected. No construction or landfilling is allowed...
Reflection: Confidence=7/10. Found key rules. I must verify the key numerical/regulatory fact before final answer.
Action: verify_answer({"statement": "Zone A of Muturajawela prohibits construction and landfilling", "source_hint": "Muturajawela Management Plan"})
Observation: [verify_answer]: Verdict: SUPPORTED | Confidence: 9/10
Reflection: Confidence=9/10. Verified. Ready to write Final Answer with citations.
Final Answer:
According to the **Muturajawela Management Plan** (p. 4):
- **Zone A** is strictly protected — construction and landfilling are prohibited [Muturajawela Management Plan, p. 4]

**Sources Used:**
- Muturajawela Management Plan, p. 4
"""

        # ── DEEP RESEARCH SYSTEM PROMPT — ADAPTIVE MODE ────────────────────
        system_prompt = f"""You are 'Marsh Thinking', an elite AI research assistant specializing in Sri Lankan wetland conservation.
Your SINGLE most important duty is ACCURACY followed by ADAPTIVE VERBOSITY. You must deliver answers that match the user's specific intent while remaining 100% grounded in the knowledge base.

═══════════════════════════════════════════════════════════════
DEEP RESEARCH PROTOCOL — FOLLOW EXACTLY
═══════════════════════════════════════════════════════════════
1. START: Call get_document_list first when unsure which document to target.
2. MULTI-HOP: Run at least 2-3 targeted searches. Never answer from a single retrieval.
3. SPECIFICITY: Search for exact terms — Act numbers, section numbers, Rs. amounts, year enacted.
4. INTENT & STRUCTURE: After gathering evidence, call **plan_answer_structure**.
   - If the question contains keywords like "informative answer", "explain", "comprehensive", "in detail" → set identified_intent="Informative", target_word_count="200-300".
   - Otherwise → identify the user's actual intent (Direct or Informative) and set word count accordingly.
{" 5. [EFFICIENT RESEARCH]: You are in 'Efficient' mode. Be concise but MUST remain 100% accurate. Do not settle for partial info." if not is_informative else " 5. [DEEP RESEARCH]: You are in 'Comprehensive' mode. Explore every secondary detail and nuance."}
 6. VERIFY: Before Final Answer, you MUST call **verify_answer** for every key numerical, legal, or policy fact. NEVER provide a final answer without verifying your most critical claims.
 7. SELF-CRITIQUE: After 4+ tool calls, list confirmed facts vs. remaining gaps before completing. Avoid all hallucinations.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — ADAPTIVE ARCHITECTURE
═══════════════════════════════════════════════════════════════
- **INTENT-DRIVEN LENGTH & ACCURACY**:
  - **Direct Intent**: (Target: 50-150 words). Provide a concise, factual answer. **ACCURACY IS PARAMOUNT**.
  - **Informative Intent**: (Target: 200-300 words). Provide a detailed analysis with headings and tables.
  - **CRITICAL**: Speed must NEVER come at the cost of correctness. If you cannot verify a fact, state that it is not found in the documents.
{"  - **EFFICIENT RESEARCH**: Research until you have the definitive answer, then verify it once and summarize concisely." if not is_informative else "  - **DEEP RESEARCH**: Perform exhaustive research across multiple documents to find every nuance before finishing."}
- **NO FIXED TEMPLATE**: Implement the custom layout you designed with `plan_answer_structure`.
- **VISUAL AIDS**: Use **TABLES** for comparing numerical data, penalties, or regulatory zones.
- **DESCRIPTIVE BULLETS**: Use **BULLET POINTS** for listing features, but follow every bullet with 2-3 sentences of explanation in Informative mode.
- **BOLDING**: **Bold** all key terms: **Act Name**, **penalty amount**, **section number**, **date**.
- **CITATIONS**: Every factual claim MUST end with its source: [Document Name, Page X].

FORMAT for reasoning:
Thought: [What information do I need and why?]
Action: tool_name({{"arg": "val"}})
Observation: [Tool result]
Reflection: [Confidence=X/10. What did I learn? What is still missing?]
...
SELF-CRITIQUE (after 4+ iterations): [Confirmed facts | Remaining gaps | Next action]
...
Final Answer:
[Accurate, intent-aware answer using your APPROVED STRUCTURE and TARGET WORD COUNT — every claim has [Source, Page]]

**Sources Used:**
- [Document, pages]

{FEW_SHOT_EXAMPLE}
Begin!
"""

        history = self.conversation_manager.get_history()
        full_context = f"SYSTEM: {system_prompt}\n\n"
        for msg in history:
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            full_context += f"{role}: {msg['content']}\n"
        full_context += f"USER: {question}\n"

        current_prompt = full_context
        # Use dynamic iterations: 10 for Fast Track, 15 for Deep Research (Informative)
        max_iterations = 15 if is_informative else 10
        iteration = 0
        last_action = None
        tool_call_count = 0
        self_critique_done = False

        try:
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"ReAct Iteration {iteration}/{max_iterations} for {self.model_name}")

                # ── Inject mandatory self-critique after 4+ tool calls ──────────
                if tool_call_count >= 4 and not self_critique_done:
                    self_critique_done = True
                    current_prompt += (
                        "\nSYSTEM INTERRUPT: You have made 4+ tool calls. "
                        "Write a SELF-CRITIQUE before continuing:\n"
                        "- List each confirmed fact with its source name [Source]\n"
                        "- List what is still missing (e.g. specific penalty values, dates)\n"
                        "- State your next specific research goal.\n"
                        "SELF-CRITIQUE: "
                    )

                # ── Final Call Nudge (Iteration 13) ───────────────────────────
                if iteration == 13:
                    current_prompt += (
                        "\nSYSTEM FINAL CALL: You are reaching the research limit. "
                        "Synthesize all gathered information now. If enough info exists, "
                        "write your Final Answer with full citations. If not, state what is missing.\n"
                    )

                # Call model with automatic rotation on quota limits
                full_llm_output = ""
                try:
                    full_llm_output = self._safe_generate_content(current_prompt)
                except Exception as e:
                    logger.error(f"Failed to generate content after rotation attempts: {e}")
                    return (
                        f"I encountered an error during research: {str(e)}. "
                        "This may happen if the content is flagged as sensitive or there is a connection issue."
                    )
                
                if not full_llm_output:
                    return "I'm sorry, I couldn't generate a response. Please try again."
                
                logger.info(f"Model response received ({len(full_llm_output)} chars)")

                lower_output = full_llm_output.lower()

                # ── Final Answer detection ──────────────────────────────────────
                if "final answer" in lower_output:
                    marker_match = re.search(r"final answer:?", full_llm_output, re.IGNORECASE)
                    if marker_match:
                        ans = full_llm_output[marker_match.end():].strip()
                        if ans:
                            logger.info(f"Final Answer found at iteration {iteration}")
                            return self._strip_react_trace(ans)

                # ── Fallback: direct output in later iterations ─────────────────
                if iteration >= 7 and not any(
                    m in lower_output for m in ["action:", "thought:", "reflection:", "self-critique"]
                ):
                    if len(full_llm_output) > 100:
                        logger.info("Assuming direct output is Final Answer (no markers, iter>=7).")
                        return self._strip_react_trace(full_llm_output.strip())

                # ── Tool call detection ─────────────────────────────────────────
                if "action" in lower_output:
                    # 1. Try strict end-anchored match first
                    action_match = re.search(
                        r"action:?\s*(\w+)\s*\((.*?)\)\s*$",
                        full_llm_output,
                        re.IGNORECASE | re.DOTALL | re.MULTILINE
                    )
                    # 2. Try greedy fallback if strict fails
                    if not action_match:
                        action_match = re.search(
                            r"action:?\s*(\w+)\s*\((.*)\)",
                            full_llm_output,
                            re.IGNORECASE | re.DOTALL
                        )

                    if action_match:
                        tool_name = action_match.group(1).strip()
                        args_str = action_match.group(2).strip()
                        observation = ""

                        try:
                            # ── Smart Argument Parsing (Handles JSON and Python-style) ─────
                            tool_args = {}
                            try:
                                clean_json = args_str
                                if '```' in clean_json:
                                    json_match = re.search(r'(\{.*\})', clean_json, re.DOTALL)
                                    if json_match: clean_json = json_match.group(1)
                                tool_args = json.loads(clean_json or "{}")
                            except (json.JSONDecodeError, ValueError):
                                # Fallback: Extract key-value pairs
                                logger.warning(f"JSON parse failed for args: {args_str}. Trying regex fallback.")
                                kv_pairs = re.findall(r'(\w+)\s*=\s*(?:"(.*?)"|\'(.*?)\'|(\d+))', args_str)
                                for k, v1, v2, v3 in kv_pairs:
                                    val = v1 or v2 or v3
                                    if v3: val = int(v3)
                                    tool_args[k] = val
                                if not tool_args and args_str:
                                    tool_args = {"query": args_str.strip("\"' ")}

                            current_action = f"{tool_name}({tool_args})"

                            if current_action == last_action:
                                observation = (
                                    "Error: You already performed this exact action. "
                                    "Try a different phrasing, call get_document_list, "
                                    "or write your Final Answer."
                                )
                            else:
                                last_action = current_action
                                tool_call_count += 1
                                try:
                                    tool_result = self.tool_executor.execute_tool(tool_name, tool_args)
                                    observation = format_tool_result_for_prompt(tool_name, tool_result)
                                    
                                    if ("No relevant documents found" in observation 
                                        or "No content found" in observation):
                                        observation += (
                                            "\n[System Hint]: No results. Try broader keywords or check document names."
                                        )
                                except Exception as e:
                                    observation = f"Error executing tool {tool_name}: {str(e)}"
                        except Exception as e:
                            logger.error(f"Action processing failed: {e}")
                            observation = f"Error: Could not process arguments '{args_str}'."

                        current_prompt += f"\n{full_llm_output}\nObservation: {observation}\n"
                        continue

                # No recognized marker — nudge the model ──────────────────────
                current_prompt += (
                    f"\n{full_llm_output}\n"
                    "Thought: I have not yet provided a Final Answer or called a tool correctly. "
                    "I must proceed with the next step in the ReAct cycle."
                )

            return (
                "The analysis required more steps than available. "
                "Please try a more specific question or break it into smaller parts."
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"

    def _safe_generate_content(self, prompt: str, max_retries: int = 5) -> str:
        """Helper to generate content with automatic API key rotation on quota limits and empty response handling"""
        retries = 0
        while retries < max_retries:
            try:
                response = self.llm_client.generate_content(prompt)
                
                # Check for empty response or safety blocks
                if not response:
                    logger.warning("Empty response object received from Gemini.")
                    return "I'm sorry, I cannot provide an answer due to an empty model response."
                
                # Success - mark key as successful if using rotator
                if self.gemini_rotator is not None and self.current_key_idx is not None:
                    self.gemini_rotator.mark_key_success(self.current_key_idx)
                
                # Safely extract text (Gemini raises ValueError if parts are empty even on finish_reason=1)
                try:
                    return response.text
                except ValueError as ve:
                    fallback = self._extract_text_from_candidates(response)
                    if fallback:
                        logger.warning("Gemini returned empty text accessor; using candidate parts as fallback.")
                        return fallback
                    logger.warning(f"Gemini returned no text parts (finish_reason={getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'none'}). Error: {ve}")
                    return "I'm sorry, I couldn't generate a response this time. Please try rephrasing."
                
            except Exception as e:
                error_str = str(e)
                # Check for quota/rate limit errors (429) or service unavailable (503/500)
                is_retryable = (
                    "429" in error_str or 
                    "503" in error_str or 
                    "500" in error_str or
                    "quota" in error_str.lower() or 
                    "limit" in error_str.lower() or
                    "unavailable" in error_str.lower()
                )
                
                if is_retryable and self.gemini_rotator:
                    msg = f"⚠️ [Marsh Thinking] API error hit for Gemini Key #{self.current_key_idx + 1}: {error_str}. Rotating key..."
                    print(f"\n{msg}")
                    logger.warning(msg)
                    
                    # Mark current key as failed (temporary)
                    if self.current_key_idx is not None:
                        self.gemini_rotator.mark_key_failed(self.current_key_idx)
                    
                    # Get next key and reconfigure
                    idx, key = self.gemini_rotator.get_next_key()
                    genai.configure(api_key=key)
                    self.current_key_idx = idx
                    
                    masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
                    print(f"🔃 [Marsh Thinking] Switched to Google Gemini API Key #{idx + 1} ({masked_key})")
                    
                    # Re-initialize the model with the new key (keep deterministic settings)
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                    self.llm_client = genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config={
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "top_k": 1,
                            "max_output_tokens": 4500,
                        },
                        safety_settings=safety_settings
                    )
                    
                    retries += 1
                    continue
                else:
                    # Some other non-quota error
                    logger.error(f"Unexpected error in _safe_generate_content: {e}")
                    raise e
        
        raise Exception("Exceeded max retries for Gemini API key rotation")

    @staticmethod
    def _extract_text_from_candidates(response) -> str:
        """Best-effort extraction when response.text raises ValueError."""
        try:
            candidates = getattr(response, "candidates", []) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content is not None else getattr(cand, "parts", []) or []
                texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
                if texts:
                    return "".join(texts)
        except Exception:
            return ""
        return ""

    @staticmethod
    def _strip_react_trace(text: str) -> str:
        """Remove any leaked ReAct chain-of-thought from the answer before returning it to the caller.
        Priority:
          1. If 'Final Answer:' appears inside the text (model repeated it), extract only what follows.
          2. Strip leading/trailing lines that start with ReAct keywords.
        """
        import re as _re
        # Pass 1 — if the model accidentally included another 'Final Answer:' marker, keep only what follows
        fa_match = _re.search(r"final answer:?", text, _re.IGNORECASE)
        if fa_match:
            candidate = text[fa_match.end():].strip()
            if candidate:
                text = candidate

        # Pass 2 — strip lines that start with known ReAct prefixes
        react_prefixes = (
            "thought:", "action:", "observation:", "reflection:",
            "self-critique:", "system interrupt:", "[system hint]",
        )
        clean_lines = []
        skip = False
        for line in text.splitlines():
            low = line.strip().lower()
            if any(low.startswith(p) for p in react_prefixes):
                skip = True
                continue
            if skip and low == "":
                skip = False
                continue
            if not skip:
                clean_lines.append(line)
        result = "\n".join(clean_lines).strip()
        return result if result else text.strip()

    def _generate_answer_with_history(self, question: str, context_docs: List[Document]) -> str:
        """Standard non-streaming generation"""
        return self._generate_answer(question, context_docs)
    def clear_conversation(self):
        self.conversation_manager.clear()

    def get_conversation_stats(self) -> Dict:
        return self.conversation_manager.get_stats()
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not self.documents:
            return {"total_chunks": 0}
        
        stats = {
            "total_chunks": len(self.documents),
            "content_types": {}
        }
        
        for doc in self.documents:
            doc_type = doc.metadata.get("type", "unknown")
            stats["content_types"][doc_type] = stats["content_types"].get(doc_type, 0) + 1
        
        return stats

    ##Chunk checker. Only for debugging purposes.
    def debug_print_chunks_for_source(self, source_name: str, max_chunks: int = 20):
        """Print all (or first N) chunks for a given PDF source."""
        matched = [d for d in self.documents if d.metadata.get("source") == source_name]
        print(f"[DEBUG] Found {len(matched)} chunks for source='{source_name}'")
        for i, doc in enumerate(matched[:max_chunks], 1):
            print(f"\n--- Chunk {i} ---")
            print("metadata:", doc.metadata)
            preview = doc.page_content[:800].replace("\n", " ")
            print("text    :", preview, "...")

# SemanticChunker is defined above (single definition only)

if __name__ == "__main__":
    """
    Manual index builder / inspector.
    Run from terminal:
        python rag_pipeline.py
    """

    PDF_FOLDER = r"C:\Users\A.Kumarasiri\OneDrive - CGIAR\WETLAND CHATBOT DOCUMENT\ALL"
    INDEX_FILE = "pdf_index_enhanced1.pkl"

    # Minimal model params – just enough to construct RAGPipeline.
    # We won't call .query() here, so the tokens/URL don't actually get used.
    model_params = {
        "llm_type": "deepseek",
        "hf_token": f'st.secrets["hf_backup_token_2"]',
        "deepseek_url": "https://router.huggingface.co/v1/chat/completions",
        "deepseek_model": "deepseek-ai/DeepSeek-R1:novita",
    }

    print("[MAIN] Initializing RAGPipeline2...")
    pipeline = RAGPipeline2(
        pdf_folder=PDF_FOLDER,
        index_file=INDEX_FILE,
        model_params=model_params,
    )

    # 1) Try to load existing index
    if pipeline.load_index():
        print("[MAIN] Existing index loaded.")
    else:
        print("[MAIN] No index found or failed to load. Building a new one...")
        try:
            total_chunks = pipeline.build_index()
            print(f"[MAIN] Index built successfully. Total chunks: {total_chunks}")
        except Exception as e:
            print("[MAIN] Index build failed:", e)
            raise
        
"""
View Chunks Script
Displays chunks from the RAG pipeline index
"""

import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================
INDEX_FILE = "pdf_index_enhanced.pkl"
TARGET_PDF = "National Environmental Policy and Strategies (2003).pdf"  # Change this to view different PDF
MAX_CHUNKS_TO_SHOW = 20

# ============================================================================
# FUNCTIONS
# ============================================================================
def load_index(index_file: str):
    """Load the pickled index and return documents"""
    print(f"Loading index from: {index_file}")
    with open(index_file, "rb") as f:
        data = pickle.load(f)
    print(f"✅ Loaded {len(data['documents'])} documents\n")
    return data["documents"]

def show_sample_chunks(documents, max_samples: int = 20):
    """Show sample chunks from different PDFs (max 1 per source PDF)"""
    if not documents:
        print("[MAIN] No documents in pipeline.documents after load/build.")
        return
    
    print(f"\n[MAIN] Showing up to {max_samples} sample chunks from different PDFs")
    print(f"       (total chunks: {len(documents)})")
    
    seen_sources = set()
    shown = 0
    
    for doc in documents:
        src = doc.metadata.get("source", "Unknown")
        if src in seen_sources:
            continue
        seen_sources.add(src)
        shown += 1
        
        print(f"\n--- Sample {shown} ---")
        print("source :", src)
        print("metadata:", doc.metadata)
        preview = doc.page_content[:400].replace("\n", " ")
        print("text    :", preview, "...")
        
        if shown >= max_samples:
            break

def debug_print_chunks_for_source(documents, source_name: str, max_chunks: int = 20):
    """Print all (or first N) chunks for a given PDF source"""
    matched = [d for d in documents if d.metadata.get("source") == source_name]
    
    print(f"\n{'='*80}")
    print(f"[MAIN] Debug: chunks for {source_name}")
    print(f"{'='*80}")
    print(f"[DEBUG] Found {len(matched)} chunks for source='{source_name}'")
    
    if not matched:
        print(f"❌ No chunks found for '{source_name}'")
        print("\nAvailable PDFs:")
        unique_sources = sorted(set(d.metadata.get("source", "Unknown") for d in documents))
        for i, src in enumerate(unique_sources[:15], 1):
            print(f"  {i:2d}. {src}")
        if len(unique_sources) > 15:
            print(f"  ... and {len(unique_sources) - 15} more")
        return
    
    print(f"Showing first {min(max_chunks, len(matched))} chunks:\n")
    
    for i, doc in enumerate(matched[:max_chunks], 1):
        print(f"--- Chunk {i} ---")
        print("source  :", doc.metadata.get("source"))
        print("metadata:", doc.metadata)
        preview = doc.page_content[:800].replace("\n", " ")
        print("text    :", preview, "...")
        print()

#

    # 2) Show a spread of sample chunks for inspection (max 1 per source PDF)
    # if pipeline.documents:
    #     print(f"\n[MAIN] Showing up to 20 sample chunks from different PDFs "
    #           f"(total chunks: {len(pipeline.documents)}):")

    #     seen_sources = set()
    #     shown = 0
    #     for doc in pipeline.documents:
    #         src = doc.metadata.get("source", "Unknown")
    #         if src in seen_sources:
    #             continue
    #         seen_sources.add(src)
    #         shown += 1

    #         print(f"\n--- Sample {shown} ---")
    #         print("source :", src)
    #         print("metadata:", doc.metadata)
    #         preview = doc.page_content[:400].replace("\n", " ")
    #         print("text    :", preview, "...")

    #         if shown >= 20:
    #             break
    # else:
    #     print("[MAIN] No documents in pipeline.documents after load/build.")

    # print("\n[MAIN] Debug: chunks for Rao-2018-Power_from_agro-waste-Business_Model_6.pdf")
    # pipeline.debug_print_chunks_for_source("Rao-2018-Power_from_agro-waste-Business_Model_6.pdf",
    #                                        max_chunks=20)
