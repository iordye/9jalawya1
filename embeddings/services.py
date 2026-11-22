from embeddings.loader import get_embedding_model, get_sparse_model
from utils.logging_config import get_logger
import numpy as np

logger = get_logger(__name__)

class EmbeddingService:

    def __init__(self):
        self.model = get_embedding_model()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Documents must be non-empty")
        try:
            logger.info("embedding documents...")
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
            )
            
        except Exception as e:
            logger.exception(f"Failed to embed document due to {e}")
            raise
        return embeddings


    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        try:
            logger.info('Embedding query...')
            embeddings = self.model.encode(
                query,
                normalize_embeddings = True 
            )
            
        except Exception as e:
            logger.exception(f"Failed to embed query due to {e}")
            raise
        return embeddings.tolist()
    
class SparseService:
    def __init__(self):
        self.model = get_sparse_model()

    def sparse_embed_doc(self, texts: list[str]):
        if not texts:
            raise ValueError("Documents must be non-empty")
        try:
            logger.info("Sparse embedding documents...")
            embeddings = list(self.model.embed(
            texts
            ))
                    
        except Exception as e:
            logger.exception(f"Failed to sparse embed document due to {e}")
            raise
        return embeddings