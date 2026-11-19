# src/ingestion/ingestor.py

# IMPORTS
import json
import time
import chromadb
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from tqdm import tqdm
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# CONSTANTS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKED_DATA_DIR = PROJECT_ROOT / "data" / "chunked"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "models\\embedding_ml\\all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "9jalawya_corpus"

# Ingestion Batch Configuration
BATCH_SIZE = 100

# EMBEDDING MODEL LOADER
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Loads the sentence-transformer embedding model.

    On first call: downloads the model from HuggingFace Hub (~80MB).
    On subsequent calls: loads from local cache (instant).

    We isolate model loading in its own function so:
    1. The model is loaded ONCE and reused across all batches
    2. If loading fails, we get a clear error before processing any data
    3. The model can be swapped out by changing EMBEDDING_MODEL_NAME

    Args:
        model_name: HuggingFace model identifier string

    Returns:
        Loaded SentenceTransformer model object
    """

    logger.info(f"Loading embedding model: {model_name}")
    logger.info("  (First run downloads and then subsequent runs loads model from cache)")

    start = time.time()

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

# CHROMA CLIENT + COLLECTION SETUP
def setup_chroma_collection(
    vectorstore_dir: Path = VECTORSTORE_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    reset: bool = False
) -> Tuple[chromadb.PersistentClient, chromadb.Collection]:
    """
    Initialises the ChromaDB persistent client and gets/creates the collection.

    ChromaDB's PersistentClient saves the database to disk automatically.
    Every time we add vectors, they are written to vectorstore_dir.
    When we restart the application, ChromaDB reloads from disk — no
    re-embedding needed.

    Args:
        vectorstore_dir: Directory where ChromaDB stores its files
        collection_name: Name of the collection to create or connect to
        reset: If True, deletes and recreates the collection (for re-ingestion)

    Returns:
        Tuple of (chroma_client, collection)
    """

    logger.info(f"Connecting to ChromaDB at: {vectorstore_dir}")
    
    client = chromadb.PersistentClient(path=str(vectorstore_dir))

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    if reset:
        # Delete the existing collection if reset is requested.
        # This is useful when you want to re-ingest after changing
        # your chunking strategy or embedding model.
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"  Existing collection '{collection_name}' deleted (reset=True)")
        except Exception:
            pass
            # If the collection doesn't exist yet, delete raises an error.
            # We silently ignore it — nothing to delete is fine.

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
        # metadata sets the distance metric for similarity search.
        # "cosine" distance measures the ANGLE between vectors
    )

    existing_count = collection.count()
    # .count() returns how many documents are currently in the collection.

    logger.info(f"  Collection '{collection_name}' ready")
    logger.info(f"  Existing documents in collection: {existing_count:,}")

    return client, collection

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

# DEDUPLICATION CHECK
def filter_already_ingested(
    chunks: List[Dict],
    collection: chromadb.Collection
) -> List[Dict]:
    """
    Filters out chunks that are already present in the ChromaDB collection.

    This enables INCREMENTAL ingestion — if you add a new document and
    re-run ingestor.py, only the new chunks are ingested. Existing chunks
    are skipped rather than duplicated.

    HOW IT WORKS:
    Each chunk has a deterministic chunk_id generated by chunker.py based
    on its content. We check which of those IDs already exist in ChromaDB
    and skip them.

    Args:
        chunks: All chunks loaded from disk
        collection: The ChromaDB collection to check against

    Returns:
        List of chunks that are NOT yet in the collection (need ingesting)
    """

    if collection.count() == 0:
        # Collection is empty — all chunks need to be ingested.
        # Skip the expensive ID lookup entirely.
        logger.info("  Collection is empty — all chunks will be ingested")
        return chunks

    logger.info("  Checking for already-ingested chunks...")

    # Extract all chunk IDs from our loaded chunks.
    all_ids = [chunk["chunk_id"] for chunk in chunks]

    # Query ChromaDB to find which of these IDs already exist.
    # We use get() with just the IDs — this is faster than a full search.
    try:
        existing = collection.get(ids=all_ids, include=[])
        # include=[] means: don't return embeddings or documents,
        # just tell us which IDs exist. Minimises data transfer.

        existing_ids = set(existing["ids"])
        # Convert the list to a set for O(1) lookup performance.
        # With 2,519 chunks, linear search through a list would be slow.
        # Set lookup is instant regardless of size.

    except Exception as e:
        logger.warning(f"  Could not check existing IDs: {e}. Will ingest all chunks.")
        return chunks

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    # List comprehension: keep only chunks whose ID is NOT in existing_ids.

    skipped = len(chunks) - len(new_chunks)
    if skipped > 0:
        logger.info(f"  Skipping {skipped:,} already-ingested chunks")
    logger.info(f"  Chunks to ingest: {len(new_chunks):,}")

    return new_chunks

# METADATA BUILDER
def build_chroma_metadata(chunk: Dict) -> Dict:
    """
    Extracts and sanitises the metadata fields for ChromaDB storage.

    ChromaDB metadata has strict requirements:
    - Values must be str, int, float, or bool ONLY
    - No None values allowed (causes silent failures)
    - No nested dictionaries or lists

    This function ensures every metadata value is a valid ChromaDB type.
    We extract only the fields we actually want to filter or display —
    not everything from the chunk dictionary.

    Args:
        chunk: A chunk dictionary from chunker.py

    Returns:
        A flat dictionary with only ChromaDB-compatible values
    """

    def safe_str(value, default: str = "") -> str:
        # Converts any value to string, returning default if None.
        # The 'or default' handles empty strings and None in one expression.
        return str(value) if value is not None else default

    def safe_int(value, default: int = 0) -> int:
        # Converts any value to int safely, returning default on failure.
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    return {
        # Document identity — used for filtered retrieval
        "doc_key":              safe_str(chunk.get("doc_key")),
        "full_document_name":   safe_str(chunk.get("full_document_name")),
        "document_category":    safe_str(chunk.get("document_category")),
        "document_year":        safe_int(chunk.get("document_year")),
        "authority":            safe_str(chunk.get("authority")),

        # Structural position — used for citation and context fetching
        "chunk_type":           safe_str(chunk.get("chunk_type")),
        "heading":              safe_str(chunk.get("heading")),
        "chunk_index":          safe_int(chunk.get("chunk_index")),
        "sub_index":            safe_int(chunk.get("sub_index")),
        "is_continuation":      bool(chunk.get("is_continuation", False)),
        # bool() converts any truthy/falsy value to True/False explicitly.

        # Size info — useful for debugging and quality monitoring
        "char_count":           safe_int(chunk.get("char_count")),
        "word_count":           safe_int(chunk.get("word_count")),
    }

# CORE INGESTION FUNCTION
def ingest_chunks(
    chunks: List[Dict],
    collection: chromadb.Collection,
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Embeds all chunks and inserts them into the ChromaDB collection.

    This is the core function that ties everything together:
    1. Splits chunks into batches
    2. Embeds each batch using the sentence-transformer model
    3. Inserts embeddings + text + metadata into ChromaDB

    Args:
        chunks: List of chunk dictionaries to ingest
        collection: The ChromaDB collection to insert into
        model: Loaded SentenceTransformer model for embedding
        batch_size: Number of chunks to process at one time

    Returns:
        Dictionary with ingestion statistics
    """

    if not chunks:
        logger.info("No chunks to ingest.")
        return {"ingested": 0, "failed": 0, "total_time": 0}

    total = len(chunks)
    ingested = 0
    failed = 0
    start_time = time.time()

    logger.info(f"Starting ingestion of {total:,} chunks in batches of {batch_size}...")

    # Calculate number of batches for the progress bar.
    num_batches = (total + batch_size - 1) // batch_size
    
    # tqdm wraps our range iterator to display a progress bar.
    # desc sets the label shown next to the bar.
    # total tells tqdm how many iterations to expect.
    for batch_num in tqdm(range(num_batches), desc="Ingesting batches", unit="batch"):

        # Calculate the start and end indices for this batch.
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total)
        # min() ensures the last batch doesn't go out of bounds.
        # e.g. for 2519 chunks, last batch: min(2500+100, 2519) = 2519

        batch = chunks[batch_start:batch_end]
        # Python list slicing: takes elements from batch_start up to
        # (but not including) batch_end.

        try:
            # --- Extract the three components ChromaDB needs ---

            ids = [chunk["chunk_id"] for chunk in batch]
            # List of unique string IDs — ChromaDB uses these as primary keys.
            # If we try to insert an ID that already exists, ChromaDB
            # raises an error (which our deduplication step prevents).

            texts = [chunk["text"] for chunk in batch]
            # List of text strings — stored as the "document" in ChromaDB.
            # This is the actual legal text returned to the LLM as context.

            metadatas = [build_chroma_metadata(chunk) for chunk in batch]
            # List of metadata dictionaries — stored alongside each vector.
            # Used for filtered retrieval and citation generation.

            # --- Generate Embeddings ---
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                # show_progress_bar=False because tqdm is already showing
                # progress at the batch level — a nested bar would be noisy.
                convert_to_numpy=True,
                # convert_to_numpy=True returns a numpy array instead of
                # a PyTorch tensor. ChromaDB expects numpy arrays or Python lists.
                normalize_embeddings=True,
                # normalize_embeddings=True scales each vector to unit length
                # (magnitude = 1). This is required for cosine similarity to
                # work correctly — without normalisation, vectors of different
                # magnitudes would distort the similarity scores.
            )
            # model.encode() returns shape (batch_size, 384) numpy array.
            # Each row is one 384-dimensional embedding vector.

            embeddings_list = embeddings.tolist()
            # .tolist() converts the numpy array to a Python list of lists.
            # ChromaDB's API accepts Python lists but not numpy arrays directly.

            # --- Insert into ChromaDB ---
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings_list,
                metadatas=metadatas,
            )
            # collection.add() inserts all four components simultaneously.
            # ChromaDB stores them together — when you query later, you get
            # the text, metadata, and similarity score back in one result.

            ingested += len(batch)

        except Exception as e:
            logger.error(f"  Batch {batch_num + 1} failed: {e}")
            failed += len(batch)
            continue
            # Log the error and continue to the next batch.
            # One failed batch should not stop the entire ingestion.

    total_time = round(time.time() - start_time, 2)
    ingestion_rate = round(ingested / total_time, 1) if total_time > 0 else 0

    logger.info(f"\nIngestion complete:")
    logger.info(f"  Successfully ingested: {ingested:,} chunks")
    logger.info(f"  Failed:                {failed:,} chunks")
    logger.info(f"  Total time:            {total_time}s")
    logger.info(f"  Ingestion rate:        {ingestion_rate} chunks/second")

    return {
        "ingested": ingested,
        "failed": failed,
        "total_time": total_time,
        "ingestion_rate": ingestion_rate,
    }

# VERIFICATION FUNCTION
def verify_ingestion(collection: chromadb.Collection) -> None:
    """
    Runs a quick sanity check after ingestion to verify everything worked.

    Performs two checks:
    1. Count check — does ChromaDB report the expected number of documents?
    2. Query check — does a test legal query return sensible results?

    This verification step is important because ChromaDB can sometimes
    silently fail to persist data (e.g. disk full, permission errors).
    Better to catch this now than during actual RAG queries.

    Args:
        collection: The ChromaDB collection that was just populated
    """

    logger.info("\nRunning ingestion verification...")

    # --- Check 1: Document Count ---
    count = collection.count()
    logger.info(f"  Total documents in ChromaDB: {count:,}")

    if count == 0:
        logger.error("  VERIFICATION FAILED: Collection is empty after ingestion!")
        return

    # --- Check 2: Test Query ---
    # Run a real semantic query against the freshly populated collection.
    # If this returns results with text, embedding and retrieval are working.
    test_queries = [
        "termination of employment contract",
        "fundamental rights of citizens",
        "company income tax assessment",
    ]

    logger.info("\n  Test query results:")

    for query in test_queries:
        try:
            results = collection.query(
                query_texts=[query],
                # query_texts: ChromaDB automatically embeds this using the
                # same embedding function we attached to the collection.
                # This is why it was important to set embedding_function
                # when creating the collection — it handles query embedding
                # transparently.

                n_results=2,
                # Return the top 2 most similar chunks.

                include=["documents", "metadatas", "distances"]
                # Specify what to include in the response:
                # "documents" → the actual text of matching chunks
                # "metadatas" → the metadata dict (doc_key, heading, etc.)
                # "distances" → the cosine distance (lower = more similar)
            )

            if results["documents"] and results["documents"][0]:
                top_result = results["documents"][0][0]
                top_meta = results["metadatas"][0][0]
                top_distance = results["distances"][0][0]
                top_similarity = round(1 - top_distance, 3)
                # Convert cosine DISTANCE to cosine SIMILARITY:
                # distance of 0.1 → similarity of 0.9 (very similar)
                # distance of 0.9 → similarity of 0.1 (very different)

                logger.info(f"\n  Query: '{query}'")
                logger.info(f"    Top match: {top_meta.get('full_document_name', 'Unknown')}")
                logger.info(f"    Section:   {top_meta.get('heading', 'Unknown')[:60]}")
                logger.info(f"    Similarity: {top_similarity}")
                logger.info(f"    Preview:   {top_result[:120]}...")
                # [:120] truncates the preview to 120 chars for readable logs.

        except Exception as e:
            logger.error(f"  Test query failed: {e}")

    logger.info("\n  Verification complete.")

# MAIN ORCHESTRATOR
def run_ingestion(reset: bool = False) -> None:
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
    logger.info("NaijaLaw AI - Ingestion Stage")
    logger.info("="*60)

    if reset:
        logger.warning("RESET MODE: Existing collection will be deleted and rebuilt.")

    # Step 1: Load the embedding model.
    model = load_embedding_model()

    # Step 2: Set up ChromaDB.
    client, collection = setup_chroma_collection(reset=reset)

    # Step 3: Load all chunks from disk.
    all_chunks = load_all_chunks()

    if not all_chunks:
        logger.error("No chunks found. Aborting ingestion.")
        return

    # Step 4: Filter out already-ingested chunks (for incremental runs).
    chunks_to_ingest = filter_already_ingested(all_chunks, collection)

    if not chunks_to_ingest:
        logger.info("All chunks are already ingested. Nothing to do.")
        logger.info("To re-ingest, run with reset=True: python ingestor.py --reset")
    else:
        # Step 5: Embed and store in ChromaDB.
        stats = ingest_chunks(chunks_to_ingest, collection, model)

    # Step 6: Verify the ingestion worked correctly.
    verify_ingestion(collection)

    logger.info("\n" + "="*60)
    logger.info("Ingestion pipeline complete.")
    logger.info(f"Vector store location: {VECTORSTORE_DIR}")
    logger.info(f"Collection name:       {CHROMA_COLLECTION_NAME}")
    logger.info(f"Total vectors stored:  {collection.count():,}")
    logger.info("Next step: run src/retrieval/retriever.py")
    logger.info("="*60)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Check if --reset flag was passed on the command line.
    # sys.argv is the list of command-line arguments.
    # sys.argv[0] is the script name, sys.argv[1] onwards are user arguments.
    # Running: python ingestor.py --reset
    # sets sys.argv = ['ingestor.py', '--reset']
    reset_flag = "--reset" in sys.argv

    run_ingestion(reset=reset_flag)
