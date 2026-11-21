import uuid, time, json
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from utils.logging_config import get_logger
from pathlib import Path
from typing import List, Dict


logger = get_logger(__name__)

client = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=120)
Collection = '9jalawya'
EMBEDDING_MODEL_NAME = "models\\embedding_ml\\all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"
CHUNKED_DATA_DIR = Path("data\chunked")
BATCH_SIZE = 256

# Collection Creation

def setup_qdrant_collection(
    collection_name: str = Collection, 
    client: QdrantClient = client, 
    reset: bool = False
) -> bool:
    """
    Set up the Qdrant collection if it doesn't exist.
    Returns True if created, False if it already existed.
    """
    logger.info(f"Setting up Qdrant collection: {collection_name}")
    created = False

    if reset and client.collection_exists(collection_name=collection_name):
        try:
            client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection {collection_name} deleted successfully(reset=True).")
        except Exception:
            pass

    if not client.collection_exists(collection_name=collection_name):
        logger.info(f"Collection {collection_name} does not exist. Creating collection...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF  # for BM25-style
                )
            },
        )
        logger.info(f"Collection {collection_name} created successfully.")
        created = True

    # Ensure payload index exists even if the collection was already created
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_key",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    return created

def load_embedding_model(
        model_name: str = EMBEDDING_MODEL_NAME
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
        logger.error("Ensure sentence-transformers is installed: pip install sentence-transformers")
        raise

def load_sparse_model(model_name: str = SPARSE_MODEL_NAME) -> SparseTextEmbedding:
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

# CHUNK LOADER
def load_all_chunks(chunked_dir: Path = CHUNKED_DATA_DIR) -> List[Dict]:
    """
    Loads all chunked JSON files and returns a flat list of all chunks.

    We flatten all chunks from all documents into one list because ChromaDB
    works with a single flat collection — it doesn't have a concept of
    sub-collections per document. Document identity is preserved through
    the doc_key metadata field on each chunk.

    Args:
        chunked_dir: Directory containing chunked JSON files

    Returns:
        Flat list of all chunk dictionaries from all documents
    """

    if not chunked_dir.exists():
        logger.error(f"Chunked data directory not found: {chunked_dir}")
        logger.error("Please run chunker.py first.")
        return []

    chunked_files = list(chunked_dir.glob("*_chunked.json"))

    if not chunked_files:
        logger.error(f"No chunked JSON files found in {chunked_dir}")
        return []

    logger.info(f"Loading chunks from {len(chunked_files)} document file(s)...")

    all_chunks = []

    for chunked_path in sorted(chunked_files):
        try:
            with open(chunked_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            chunks = doc_data.get("chunks", [])
            all_chunks.extend(chunks)
            # .extend() adds all items from chunks into all_chunks.
            # This is more efficient than calling .append() in a loop
            # because it adds all items in one operation.

            logger.info(
                f"  {doc_data.get('doc_key', 'unknown'):<35} "
                f"{len(chunks):>4} chunks loaded"
            )

        except Exception as e:
            logger.error(f"Failed to load {chunked_path.name}: {e}")
            continue

    logger.info(f"  Total chunks loaded: {len(all_chunks):,}")
    return all_chunks

def ingest(chunks, batch_size=BATCH_SIZE, dense_model=None, sparse_model=None, client=client):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        dense = dense_model.encode(texts, normalize_embeddings=True)
        sparse = list(sparse_model.embed(texts))

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])),
                vector={
                    "dense": d.tolist(),
                    "sparse": models.SparseVector(
                        indices=s.indices.tolist(), values=s.values.tolist()
                    ),
                },
                payload={
                    "doc_key": c["doc_key"],
                    "section": c["section_number_int"],
                    "section_title": c["section_title"],
                    "text": c["text"],
                    "corpus_version": "2026-11",
                },
            )
            for c, d, s in zip(batch, dense, sparse)
        ]
        client.upsert(Collection, points=points)

def run_injestion(reset: bool= False) -> None:
    
    """
          Full ingestion pipeline orchestrator.
        
            Runs all four steps in sequence:
            1. Load embedding model
            2. Set up ChromaDB collection
            3. Load and deduplicate chunks
            4. Embed and ingest chunks
            5. Verify results
        
            Args:
                reset: If True, clears the existing collection before ingesting.
                       Use this when you have changed your chunking strategy and
                       want to rebuild the vector store from scratch.
                       Default False for safe incremental ingestion.
    """

    logger.info("="*60)
    logger.info("9jalawya - Ingestion Stage")
    logger.info("="*60)

    if reset:
       logger.warning("RESET MODE: Existing collection will be deleted and rebuilt.")

    setup_qdrant_collection(collection_name=Collection, client=client, reset=reset)
    embedding = load_embedding_model()
    sparse = load_sparse_model()
    chunks = load_all_chunks()
    ingest(chunks, batch_size=BATCH_SIZE, dense_model=embedding, sparse_model=sparse, client=client)
    logger.info("Ingestion complete. Total chunks ingested: {:,}".format(len(chunks)))


if __name__ == "__main__":
    import sys
    reset_flag = "--reset" in sys.argv
    run_injestion(reset=reset_flag)


