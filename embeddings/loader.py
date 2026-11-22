from sentence_transformers import SentenceTransformer
from utils.logging_config import get_logger
from functools import lru_cache
from fastembed import SparseTextEmbedding
from config.settings import Settings
import time 

logger = get_logger(__name__)

settings = Settings()

@lru_cache(maxsize=1)
def get_embedding_model(
        model_name: str = settings.embeddings.model_name
        ) -> SentenceTransformer:
    """
    Load the embedding model.
    Args:
        model_name: HuggingFace model identifier string
    
    Returns:
        Loaded SentenceTransformer model object
    """
    start = time.time()
    logger.info(f"Loading embedding model: {model_name}")
    try:
            model = SentenceTransformer(model_name)
            elapsed = round(time.time() - start, 2)
            logger.info(f"  Model loaded successfully in {elapsed}s")
            logger.info(f"  Embedding dimensions: {model.get_sentence_embedding_dimension()}")
           
            return model
    
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}")
        raise

@lru_cache(maxsize=1)
def load_sparse_model(model_name: str = settings.embeddings.model_name) -> SparseTextEmbedding:
    """
    Load the sparse embedding model (BM25).
    
    Returns:
        Loaded SparseTextEmbedding model object
    """
    start = time.time()
    logger.info("Loading sparse embedding model: Qdrant/bm25")
    try:
        sparse_model = SparseTextEmbedding(model_name)
        elapsed = round(time.time() - start, 2)
        logger.info(f"  Sparse model loaded successfully in {elapsed}s")
        return sparse_model
    
    except Exception as e:
        logger.error(f"Failed to load sparse embedding model: {e}")
        raise

    
