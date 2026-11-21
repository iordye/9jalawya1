from sentence_transformers import CrossEncoder
from utils.logging_config import get_logger
from typing import Protocol

logger = get_logger(__name__)

class Reranker(Protocol):
    def rerank(self, query: str, hits: list, top_n: int) -> list[tuple[str, float]]: ...

class MiniLMranker:
    def __init__(self):
        self.model = CrossEncoder("models\reranker_ml\ms-marco-MiniLM-L-6-v2")

    def rerank(
            self, query: str, 
            hits: list, 
            top_n: int
            ) -> list[tuple[str, float]]:
        pass