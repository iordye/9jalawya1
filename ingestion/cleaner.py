# src/ingestion/cleaner.py
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import time
from collections import Counter
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# CONSTANTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXTRACTED_DATA_DIR = PROJECT_ROOT / "data" / "extracted"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
HEADER_DETECTION_THRESHOLD = 0.55
MIN_LINE_LENGTH = 3
MIN_PAGE_CHAR_COUNT = 100
MAX_HEADER_LENGTH = 80             # headers are usually short


# STEP 1: UNICODE NORMALISATION
def normalise_unicode(text: str) -> str:
    """
    Converts special Unicode characters to their standard equivalents.
    Args:
        text: Raw text string from PDF extraction

    Returns:
        Text with normalised Unicode characters
    """

    replacements = {
        "\u2013": "-",   # en dash → hyphen
        "\u2014": "-",   # em dash → hyphen
        "\u2018": "'",   # left single quote → apostrophe
        "\u2019": "'",   # right single quote → apostrophe
        "\u201c": '"',   # left double quote → straight quote
        "\u201d": '"',   # right double quote → straight quote
        "\u00a0": " ",   # non-breaking space → regular space
        "\u00b7": ".",   # middle dot → period
        "\u2022": "-",   # bullet point → hyphen
        "\u00a7": "S.",  # section symbol § → "S."
    }

    text = unicodedata.normalize("NFKD", text)

    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)

    text = text.encode("ascii", "ignore").decode("ascii")  

    return text

# STEP 2: HEADER AND FOOTER DETECTION

def detect_repeated_lines(pages: List[Dict]) -> set:
    """
    Analyses all pages to find lines that repeat suspiciously often.
    These are almost certainly page headers, footers, or watermarks.

    Strategy:
    - Split every page into individual lines
    - Count how many pages each unique line appears on
    - Flag lines that appear on >= HEADER_DETECTION_THRESHOLD of all pages

    Args:
        pages: List of page dictionaries from the extracted JSON

    Returns:
        A set of strings that should be treated as headers/footers and removed.
    """

    total_pages = len(pages)
    if total_pages == 0:
        return set()

    line_page_counts = Counter()

    for page in pages:
        text = page.get("text", "")
        lines_on_page = set(
            line.strip() for line in text.split("\n")
            if line.strip() and len(line.strip()) > MIN_LINE_LENGTH
        )

        for line in lines_on_page:
            line_page_counts[line] += 1
            
    # Identify lines that appear on too many pages — these are headers/footers.
    repeated_lines = set()
    for line, count in line_page_counts.items():
        frequency = count / total_pages

        if frequency >= HEADER_DETECTION_THRESHOLD:
            repeated_lines.add(line)
            logger.debug(f"  Header/footer detected ({frequency:.0%} of pages): '{line[:60]}'")
            
    logger.info(f"  Detected {len(repeated_lines)} repeated header/footer line(s)")
    logger.info(f"Repeated lines: {repeated_lines}")
    return repeated_lines

# STEP 3: PAGE-LEVEL CLEANING FUNCTION

def clean_page_text(text: str, repeated_lines: set) -> str:
    """
    Applies all cleaning operations to a single page's text.

    Args:
        text: Raw text from one page
        repeated_lines: Set of header/footer lines to remove

    Returns:
        Cleaned text string
    """

    if not text or not text.strip():
        return ""
        
    # CLEANING OPERATION 1: Unicode Normalisation     
    text = normalise_unicode(text)

    # CLEANING OPERATION 2: Remove Repeated Headers/Footers
    lines = text.split("\n")

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        
        if stripped not in repeated_lines:
            cleaned_lines.append(line)


    text = "\n".join(cleaned_lines)
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    
    text = re.sub(
        r'^\s*(?:M{0,4})(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\s*$',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    # CLEANING OPERATION 5: Fix Broken Hyphenated Words 
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # CLEANING OPERATION 6: Fix Spaced-Out Characters
    text = re.sub(r'\b([A-Z])\s(?=[A-Z]\s)', r'\1', text)

    # CLEANING OPERATION 7: Normalise Whitespace Within Lines
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # CLEANING OPERATION 8: Normalise Excessive Blank Lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # CLEANING OPERATION 9: Strip Leading/Trailing Whitespace
    text = text.strip()

    return text

# STEP 4: DOCUMENT-LEVEL CLEANING ORCHESTRATOR

def clean_document(extracted_data: Dict) -> Optional[Dict]:
    """
    Cleans all pages of a single extracted document.

    This function orchestrates the full cleaning pipeline for one document:
    1. Detects headers/footers across all pages
    2. Cleans each page individually
    3. Filters out pages that are still too sparse after cleaning
    4. Returns a cleaned document dictionary ready for chunking

    Args:
        extracted_data: Dictionary from pdf_extractor.py containing pages + metadata

    Returns:
        Cleaned document dictionary, or None if cleaning fails
    """

    doc_key = extracted_data.get("doc_key", "unknown")
    pages = extracted_data.get("pages", [])
    metadata = extracted_data.get("metadata", {})

    if not pages:
        logger.error(f"No pages found in extracted data for '{doc_key}'")
        return None

    logger.info(f"Cleaning document: {doc_key} ({len(pages)} pages)")
    start_time = time.time()

    # Step A: Detect repeated headers/footers across all pages.
    logger.info("  Detecting headers and footers...")
    repeated_lines = detect_repeated_lines(pages)

    # Step B: Clean each page individually.
    cleaned_pages = []
    skipped_pages = []

    for page in pages:
        page_number = page.get("page_number")
        raw_text = page.get("text", "")

        # Apply all cleaning operations to this page's text.
        cleaned_text = clean_page_text(raw_text, repeated_lines)

        # Check if the cleaned page has enough content to be useful.
        if len(cleaned_text) < MIN_PAGE_CHAR_COUNT:
            logger.debug(f"  Page {page_number}: too sparse after cleaning ({len(cleaned_text)} chars), skipping")
            skipped_pages.append(page_number)
            continue

        # Build the cleaned page dictionary.
        cleaned_page = {
            "page_number": page_number,
            "text": cleaned_text,
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "original_char_count": page.get("char_count", 0)
        }

        cleaned_pages.append(cleaned_page)

    elapsed = round(time.time() - start_time, 2)

    # Calculate cleaning statistics for quality reporting.
    original_total_chars = sum(p.get("char_count", 0) for p in pages)
    

    cleaned_total_chars = sum(p["char_count"] for p in cleaned_pages)

    noise_removed_pct = 0
    if original_total_chars > 0:
        noise_removed_pct = round(
            (1 - cleaned_total_chars / original_total_chars) * 100, 1
        )

    logger.info(
        f"  Cleaning complete: {len(cleaned_pages)}/{len(pages)} pages kept | "
        f"{noise_removed_pct}% noise removed | {elapsed}s"
    )

    if skipped_pages:
        logger.info(f"  Skipped sparse pages: {skipped_pages}")

    # Warn if we removed an unusually large amount of content.
    # More than 40% removed might indicate the threshold is too aggressive.
    if noise_removed_pct > 40:
        logger.warning(
            f"  High noise removal ({noise_removed_pct}%) for '{doc_key}'. "
            f"Review cleaned output to ensure legal content is preserved."
        )

    # Build and return the cleaned document dictionary.
    return {
        "doc_key": doc_key,
        "metadata": {
            **metadata,
            "cleaned_pages": len(cleaned_pages),
            "skipped_pages": len(skipped_pages),
            "original_total_chars": original_total_chars,
            "cleaned_total_chars": cleaned_total_chars,
            "noise_removed_percent": noise_removed_pct,
            "cleaning_time_seconds": elapsed,
        },
        "pages": cleaned_pages,
    }

# SAVE FUNCTION

def save_cleaned_document(cleaned_data: Dict, output_dir: Path) -> Path:
    """
    Saves the cleaned document to a JSON file.

    Args:
        cleaned_data: The dictionary returned by clean_document()
        output_dir: The folder to save the JSON file into

    Returns:
        Path to the saved JSON file
    """

    output_path = output_dir / f"{cleaned_data['doc_key']}_cleaned.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    logger.info(f"  Saved to: {output_path.name}")
    return output_path


# BATCH PROCESSING FUNCTION

def clean_all_documents(
    extracted_dir: Path = EXTRACTED_DATA_DIR,
    output_dir: Path = CLEANED_DATA_DIR
) -> List[Dict]:
    """
    Cleans ALL extracted JSON files found in the extracted data directory.

    This mirrors the structure of extract_all_documents() in pdf_extractor.py —
    each stage in our pipeline has a batch function that processes all documents
    automatically. This consistency makes the pipeline easy to understand and extend.

    Args:
        extracted_dir: Folder containing extracted JSON files
        output_dir: Folder to save cleaned JSON files

    Returns:
        List of all successfully cleaned document dictionaries
    """

    if not extracted_dir.exists():
        logger.error(f"Extracted data directory not found: {extracted_dir}")
        logger.error("Please run pdf_extractor.py file first.")
        return []

    # Find all extracted JSON files.
    extracted_files = list(extracted_dir.glob("*_extracted.json"))

    if not extracted_files:
        logger.error(f"No extracted JSON files found in {extracted_dir}")
        logger.error("Please run pdf_extractor.py file first to generate extracted files.")
        return []

    logger.info(f"Found {len(extracted_files)} extracted document(s) to clean:")
    for f in extracted_files:
        logger.info(f"  - {f.name}")

    all_cleaned = []

    for extracted_path in sorted(extracted_files):

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {extracted_path.name}")
        logger.info(f"{'='*60}")

        # Load the extracted JSON file.
        try:
            with open(extracted_path, "r", encoding="utf-8") as f:
                extracted_data = json.load(f)
                # json.load() reads a JSON file and converts it back into
                # a Python dictionary. This is the inverse of json.dump().
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {extracted_path.name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to read {extracted_path.name}: {e}")
            continue

        # Clean the document.
        cleaned_data = clean_document(extracted_data)

        if cleaned_data is None:
            logger.error(f"Cleaning failed for {extracted_path.name}")
            continue

        # Save the cleaned output.
        save_cleaned_document(cleaned_data, output_dir)
        all_cleaned.append(cleaned_data)

    # Final summary.
    logger.info(f"\n{'='*60}")
    logger.info(f"CLEANING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total documents processed: {len(extracted_files)}")
    logger.info(f"Successfully cleaned:      {len(all_cleaned)}")
    logger.info(f"Failed:                    {len(extracted_files) - len(all_cleaned)}")

    # Print per-document noise removal stats.
    logger.info(f"\nNoise removal per document:")
    for doc in all_cleaned:
        meta = doc.get("metadata", {})
        logger.info(
            f"  {doc['doc_key']:<35} "
            f"{meta.get('noise_removed_percent', 0)}% noise removed | "
            f"{meta.get('cleaned_pages', 0)} pages kept"
        )
        # :<35 is a format specifier — left-align the text in a field 35
        # characters wide. This aligns all the percentages in a neat column.

    return all_cleaned

# ENTRY POINT

if __name__ == "__main__":
    logger.info("NaijaLaw AI - Text Cleaning Stage")
    logger.info(f"Input directory:  {EXTRACTED_DATA_DIR}")
    logger.info(f"Output directory: {CLEANED_DATA_DIR}")

    results = clean_all_documents()

    if results:
        logger.info(f"\nCleaning complete. {len(results)} document(s) ready for chunking.")
        logger.info("Next step: run src/ingestion/chunker.py")
    else:
        logger.error("Cleaning produced no results. Check extracted files and paths.")
