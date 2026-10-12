import logging
from typing import List, Optional
import numpy as np
from core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton service for generating prompt embeddings using SentenceTransformer.
    Input: Text query
    Output: 384-dimensional normalized vector (matching trained TF evaluator models)
    """
    _instance: Optional["EmbeddingService"] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.model = None
        self._load_model()

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = EmbeddingService()
        return cls._instance

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Fallback to numpy embeddings if needed.")
            self.model = None

    def _embed_via_hf_api(self, texts: List[str]) -> np.ndarray:
        """
        Fallback embedding generation using Hugging Face Inference API.
        Queries sentence-transformers model via HF Feature Extraction endpoint.
        """
        import requests

        hf_token = settings.HUGGINGFACE_API_KEY
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }

        logger.info(f"Invoking Hugging Face Inference API fallback for model: {self.model_name}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)

        if response.status_code != 200:
            error_msg = f"Hugging Face Inference API error ({response.status_code}): {response.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        data = response.json()
        vecs = np.array(data, dtype=np.float32)

        # Handle dimension edge-cases (e.g. token-level output [batch, seq_len, dim])
        if vecs.ndim == 3:
            vecs = np.mean(vecs, axis=1)
        elif vecs.ndim == 1:
            vecs = np.expand_dims(vecs, axis=0)

        # L2-normalize to 384-dimensional unit vector matching trained TF evaluator models
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        normalized_vecs = vecs / (norms + 1e-8)
        return normalized_vecs.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generates a 384-dimensional query embedding.
        Uses local SentenceTransformer if loaded; otherwise falls back to Hugging Face API.
        """
        if self.model is not None:
            emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            return emb[0].astype(np.float32)
        else:
            batch_result = self._embed_via_hf_api([text])
            return batch_result[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a batch of strings into a (N, 384) array.
        Uses local SentenceTransformer if loaded; otherwise falls back to Hugging Face API.
        """
        if self.model is not None:
            emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return emb.astype(np.float32)
        else:
            return self._embed_via_hf_api(texts)


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService.get_instance()


def generate_query_embedding(query: str) -> np.ndarray:
    """
    Convenience function: Generates a 384-dimensional query embedding
    using SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').
    """
    return get_embedding_service().embed_text(query)
