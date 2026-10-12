import os
import uuid
import logging
from typing import Dict, Optional, Tuple
import numpy as np
from core.config import settings

logger = logging.getLogger(__name__)


def connect_vector_db():
    """
    Connects to the strictly local persistent Vector DB (Qdrant).
    Uses local disk storage (or local in-memory fallback) exclusively.
    No external or remote API connections are permitted.
    Creates or reuses local collection with 384-dim COSINE distance.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    storage_path = settings.QDRANT_STORAGE_PATH
    os.makedirs(storage_path, exist_ok=True)
    logger.info(f"Initializing strictly local embedded Qdrant disk storage at: {storage_path}")
    client = QdrantClient(path=storage_path)

    # Ensure collection exists
    collections = [col.name for col in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Created persistent local Qdrant collection: {settings.QDRANT_COLLECTION}")
    return client


class SemanticCacheService:
    """
    Semantic Vector Caching service powered by local persistent Qdrant storage.
    Matches incoming prompt embeddings against cached responses to eliminate redundant LLM calls.
    """
    _instance: Optional["SemanticCacheService"] = None

    def __init__(self):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION
        self.similarity_threshold = settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD
        self._init_qdrant()

    @classmethod
    def get_instance(cls) -> "SemanticCacheService":
        if cls._instance is None:
            cls._instance = SemanticCacheService()
        return cls._instance

    def _init_qdrant(self):
        try:
            self.client = connect_vector_db()
        except Exception as e:
            logger.warning(f"Could not initialize local Qdrant ({e}). Falling back to in-memory mode.")
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Distance, VectorParams
                self.client = QdrantClient(location=":memory:")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
                )
            except Exception as in_mem_err:
                logger.error(f"In-memory Qdrant fallback also failed: {in_mem_err}")
                self.client = None

    def lookup(
        self,
        prompt_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[Dict, float]]:
        """
        Searches for a similar query with confidence score >= threshold.
        If found: returns (payload, score).
        If not found: returns None to proceed to model selection.
        """
        if self.client is None:
            return None

        effective_threshold = threshold if threshold is not None else self.similarity_threshold
        query_vector = prompt_embedding.tolist() if isinstance(prompt_embedding, np.ndarray) else prompt_embedding

        try:
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=1,
                score_threshold=effective_threshold
            )

            if search_results:
                top_hit = search_results[0]
                logger.info(f"Semantic Cache Hit! Similarity: {top_hit.score:.4f} >= threshold ({effective_threshold})")
                return top_hit.payload, float(top_hit.score)

        except Exception as e:
            logger.error(f"Error querying local semantic cache: {e}")

        return None

    def store(
        self,
        prompt: str,
        prompt_embedding: np.ndarray,
        response_text: str,
        selected_model: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Persists prompt embedding and response payload into local Qdrant storage.
        """
        if self.client is None:
            return False

        vector = prompt_embedding.tolist() if isinstance(prompt_embedding, np.ndarray) else prompt_embedding
        point_id = str(uuid.uuid4())
        payload = {
            "prompt": prompt,
            "response_text": response_text,
            "selected_model": selected_model,
            "metadata": metadata or {}
        }

        try:
            from qdrant_client.http.models import PointStruct
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            logger.info(f"Saved successful query to Vector DB [id={point_id}]")
            return True
        except Exception as e:
            logger.error(f"Failed to store in local semantic cache: {e}")
            return False

    def clear(self) -> bool:
        """Clears and resets the local semantic cache collection."""
        if self.client is None:
            return False
        try:
            from qdrant_client.http.models import Distance, VectorParams
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Reset local Qdrant collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing local cache: {e}")
            return False


def get_semantic_cache() -> SemanticCacheService:
    return SemanticCacheService.get_instance()


def search_similar_query(query_embedding: np.ndarray, threshold: Optional[float] = None) -> Optional[Tuple[Dict, float]]:
    """
    Searches for a similar query. If confidence >= threshold, returns (payload, score).
    Otherwise returns None.
    """
    return get_semantic_cache().lookup(query_embedding, threshold=threshold)


def save_successful_query(prompt: str, response: str, query_embedding: np.ndarray, selected_model: str = "routed_llm") -> bool:
    """
    Saves successful query, response, and embedding to Vector DB.
    """
    return get_semantic_cache().store(
        prompt=prompt,
        prompt_embedding=query_embedding,
        response_text=response,
        selected_model=selected_model
    )
