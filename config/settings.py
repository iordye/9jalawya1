# config/settings.py

from dataclasses import dataclass, field
import os

@dataclass
class EmbeddingConfig:
    """
    """

    model_name: str = "models\\embedding_ml\\all-MiniLM-L6-v2"
    batch_size: int = 256

@dataclass
class SparseConfig:
    """
    """

    model_name: str = "Qdrant/bm25"


@dataclass
class Settings:
    """
    """

    embeddings: EmbeddingConfig = field(default_factory = EmbeddingConfig)
    sparse: SparseConfig = field(default_factory = SparseConfig)
