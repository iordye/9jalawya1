# config/settings.py

from dataclasses import dataclass, field
import os
from qdrant_client import QdrantClient

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
class VectorStore:

    client: QdrantClient = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=120) 
    collection_name = '9jalawya'


@dataclass
class Settings:
    """
    """

    embeddings: EmbeddingConfig = field(default_factory = EmbeddingConfig)
    sparse: SparseConfig = field(default_factory = SparseConfig)
    vectorstore: VectorStore = field(default_factory = VectorStore)
    
