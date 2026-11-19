# src/ingestion/pdf_extractor.py
import os
import json
from pathlib import Path
import pdfplumber
from typing import Dict, List, Optional
import logging
import time
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# CONSTANTS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DATA_DIR = PROJECT_ROOT / "data" / "extracted"
EXTRACTED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Legal document registry — maps filename to human-readable metadata.
# This is important because filenames like "cap_l1_lfn2004.pdf" are not
# meaningful. By registering each document, we can attach rich metadata
# to every extracted chunk later. This metadata travels with the text
# all the way into the vector store and is used for filtered retrieval.
LEGAL_DOCUMENTS = {
    "constitution": {
        "full_name": "Constitution of the Federal Republic of Nigeria 1999 (as amended)",
        "category": "constitution",
        "year": 1999,
        "authority": "Federal Republic of Nigeria",
    },
    "labour_act": {
        "full_name": "Labour Act Cap L1 LFN 2004",
        "category": "act",
        "year": 2004,
        "authority": "Federal Republic of Nigeria",
    },
    "companies_income_tax_act": {
        "full_name": "Companies Income Tax Act (CITA)",
        "category": "act",
        "year": 2004,
        "authority": "Federal Inland Revenue Service",
    },
    "criminal_code_act": {
        "full_name": "Criminal Code Act Cap C38 LFN 2004",
        "category": "act",
        "year": 2004,
        "authority": "Federal Republic of Nigeria",
    },
    "CRIMINAL JUSTICE (MISCELLANEOUS PROVISIONS) ACT": {
        "full_name": "CRIMINAL JUSTICE (MISCELLANEOUS PROVISIONS) ACT",
        "category": "act",
        "year": 1975,
        "authority": "Federal Republic of Nigeria",
    },
    "2015_Administration_of_Criminal_Justice_Act": {
        "full_name": "2015 Administration of Criminal Justice Act",
        "category": "act",
        "year": 2015,
        "authority": "Federal Republic of Nigeria",
    },"BIRTHS, DEATHS, ETC. (COMPULSORY REGISTRATION) ACT": {
        "full_name": "BIRTHS, DEATHS, ETC. (COMPULSORY REGISTRATION) ACT",
        "category": "act",
        "year": 1992,
        "authority": "Federal Republic of Nigeria",
    },"CHILD'S RIGHT ACT": {
        "full_name": "CHILD'S RIGHT ACT",
        "category": "act",
        "year": 2003,
        "authority": "Federal Republic of Nigeria",
    },"CONSUMER PROTECTION COUNCIL ACT": {
        "full_name": "CONSUMER PROTECTION COUNCIL ACT",
        "category": "act",
        "year": 2004,
        "authority": "Federal Republic of Nigeria",
    },"electoral_act": {
        "full_name": "Electoral Act 2022",
        "category": "act",
        "year": 2022,
        "authority": "Federal Republic of Nigeria",
    },"CRIMINAL PROCEDURE (NORTHERN STATES) ACT": {
        "full_name": "CRIMINAL PROCEDURE (NORTHERN STATES) ACT",
        "category": "act",
        "year": 1975,
        "authority": "Federal Republic of Nigeria",
    },"CRIMINAL PROCEDURE ACT": {
        "full_name": "CRIMINAL PROCEDURE ACT",
        "category": "act",
        "year": 1990,
        "authority": "Federal Republic of Nigeria",
    },"cybercrime": {
        "full_name": "Cyber Crime act",
        "category": "act",
        "year": 2015,
        "authority": "Federal Republic of Nigeria",
    },"freedom-of-information-act": {
        "full_name": "Freedom of Information Act",
        "category": "act",
        "year": 2011,
        "authority": "Federal Republic of Nigeria",
    },"LAND USE ACT": {
        "full_name": "LAND USE ACT",
        "category": "act",
        "year": 1978,
        "authority": "Federal Republic of Nigeria",
    },"LANDS (TITLE VESTING, ETC.) ACT": {
        "full_name": "LANDS (TITLE VESTING, ETC.) ACT",
        "category": "act",
        "year": 1975,
        "authority": "Federal Republic of Nigeria",
    },"LEGAL AID ACT": {
        "full_name": "LEGAL AID ACT",
        "category": "act",
        "year": 2011,
        "authority": "Federal Republic of Nigeria",
    },"MARRIAGE ACT": {
        "full_name": "MARRIAGE ACT",
        "category": "act",
        "year": 1990,
        "authority": "Federal Republic of Nigeria",
    },"MATRIMONIAL CAUSES ACT": {
        "full_name": "MATRIMONIAL CAUSES ACT",
        "category": "act",
        "year": 1970,
        "authority": "Federal Republic of Nigeria",
    }
}


def extract_text_from_pdf(pdf_path: Path) -> Optional[Dict]:
    """
    Extracts raw text from a single PDF file, page by page.

    This function opens the PDF, reads each page, extracts its text content,
    and returns a structured dictionary containing the raw text, page count,
    and basic metadata. 

    Args:
        pdf_path (Path): The absolute path to the PDF file to extract from.

    Returns:
        Dict containing extracted pages and metadata, or None if extraction fails.
    """

    # Validate that the file actually exists before trying to open it.
    # This gives a clear, helpful error instead of a confusing library crash.
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return None
        
    # Get the filename without extension (e.g. "labour_act" from "labour_act.pdf")
    doc_key = pdf_path.stem
    
    # Look up metadata for this document from our registry.
    doc_metadata = LEGAL_DOCUMENTS.get(doc_key, {})
    if not doc_metadata:
        logger.warning(
            f"'{doc_key}' not found in LEGAL_DOCUMENTS registry. "
            f"Extraction will proceed but metadata will be empty. "
            f"Consider adding it to the LEGAL_DOCUMENTS dictionary."
        )
        
    logger.info(f"Starting extraction: {pdf_path.name}")
    start_time = time.time()

    pages_data = []
    total_pages = 0
    failed_pages = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:

            total_pages = len(pdf.pages)
            logger.info(f"Total pages detected: {total_pages}")

            for i, page in enumerate(pdf.pages):
                page_number = i + 1

                try:
                    raw_text = page.extract_text(layout=True)

                    if not raw_text or not raw_text.strip():
                        logger.debug(f"  Page {page_number}: empty or non-text content, skipping")
                        failed_pages.append(page_number)
                        continue
                       
                    page_data = {
                        "page_number": page_number,
                        "text": raw_text,
                        "char_count": len(raw_text),
                        "word_count": len(raw_text.split())
                    }

                    pages_data.append(page_data)
                    

                except Exception as page_error:
                    logger.warning(f"  Page {page_number} extraction failed: {page_error}")
                    failed_pages.append(page_number)
                    continue

    except Exception as e:
        logger.error(f"Failed to open or process {pdf_path.name}: {e}")
        return None

    # Calculate how long extraction took.
    elapsed = round(time.time() - start_time, 2)
    successful_pages = len(pages_data)

    logger.info(
        f"  Extraction complete: {successful_pages}/{total_pages} pages extracted "
        f"in {elapsed}s | Failed pages: {failed_pages if failed_pages else 'None'}"
    )

    # If we got absolutely nothing, return None to signal total failure.
    if successful_pages == 0:
        logger.error(f"No text could be extracted from {pdf_path.name}. "
                     f"The file may be scanned/image-based. Consider OCR processing.")
        return None

    # Build and return the final result dictionary for this document.
    # This is the object that gets saved to JSON and passed to the cleaning stage.
    return {
        "doc_key": doc_key,
        # The filename stem — used as a unique identifier for this document.

        "metadata": {
            **doc_metadata,
            "source_file": pdf_path.name,
            "total_pages": total_pages,
            "extracted_pages": successful_pages,
            "failed_pages": failed_pages,
            "extraction_time_seconds": elapsed,
        },
        "pages": pages_data,
        # The list of page dictionaries — the actual extracted text content.
    }


# SAVE FUNCTION

def save_extracted_document(extracted_data: Dict, output_dir: Path) -> Path:
    """
    Saves the extracted document data to a JSON file.

   Args:
        extracted_data: The dictionary returned by extract_text_from_pdf()
        output_dir: The folder to save the JSON file into

    Returns:
        Path to the saved JSON file
    """

    output_path = output_dir / f"{extracted_data['doc_key']}_extracted.json"

    with open(output_path, "w", encoding="utf-8") as f:
        # open() in "w" mode creates or overwrites the file.
        # encoding="utf-8" is critical — Nigerian legal documents may contain
        # special characters (section symbols §, em dashes —, etc.) that only
        # UTF-8 handles correctly. Without this, you get encoding errors or
        # silent character corruption.

        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        # json.dump() serialises (converts) our Python dictionary to JSON text
        # and writes it to the file.
        # ensure_ascii=False: allows non-ASCII characters (like § or —) to be
        # written as themselves rather than ugly escape sequences (\u00a7).
        # indent=2: makes the JSON human-readable with 2-space indentation.
        # Without indent, the entire file is one unreadable line.

    logger.info(f"  Saved to: {output_path.name}")
    return output_path


# BATCH PROCESSING FUNCTION

def extract_all_documents(raw_dir: Path = RAW_DATA_DIR,
                           output_dir: Path = EXTRACTED_DATA_DIR) -> List[Dict]:
    """
    Processes ALL PDF files found in the raw data directory.

    This is the main orchestration function. Instead of calling
    extract_text_from_pdf() manually for each file, this function
    finds all PDFs automatically and processes them in a loop.
    This is important for our project because as we add more legal
    documents later, this function handles them automatically with
    no code changes needed.

    Args:
        raw_dir: Folder containing the raw PDF files (default: data/raw/)
        output_dir: Folder to save extracted JSON files (default: data/extracted/)

    Returns:
        List of all successfully extracted document dictionaries
    """

    # Validate that the raw data directory exists.
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        logger.error("Please create the folder and place your PDF files inside it.")
        return []

    # Find all PDF files in the directory.
    # .glob("*.pdf") returns a generator of all files matching the pattern.
    # list() converts the generator to a list so we can count and iterate.
    # The pattern is case-sensitive on Linux/Mac, so we also check uppercase.
    pdf_files = list(raw_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in {raw_dir}")
        logger.error("Please place your legal PDF documents in the data/raw/ folder.")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process:")
    for f in pdf_files:
        logger.info(f"  - {f.name}")

    # Process each PDF file one by one.
    all_extracted = []
    
    for pdf_path in sorted(pdf_files):
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'='*60}")

        extracted = extract_text_from_pdf(pdf_path)
        # Call our core extraction function for this PDF.

        if extracted is None:
            logger.error(f"Skipping {pdf_path.name} due to extraction failure.")
            continue

        saved_path = save_extracted_document(extracted, output_dir)

        all_extracted.append(extracted)

    # Final summary report across all documents.
    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total documents processed: {len(pdf_files)}")
    logger.info(f"Successfully extracted:    {len(all_extracted)}")
    logger.info(f"Failed:                    {len(pdf_files) - len(all_extracted)}")
    logger.info(f"Output directory:          {output_dir}")

    return all_extracted

# ENTRY POINT

if __name__ == "__main__":
    # This block only runs when the script is executed DIRECTLY:
    #   python src/ingestion/pdf_extractor.py
    # It does NOT run when this file is imported by another module.
    # This is a Python best practice — it allows the functions to be reused
    # elsewhere (like by main.py or a notebook) without triggering extraction.

    logger.info("9jaLawya AI — PDF Extraction Stage")
    logger.info(f"Raw PDF directory:  {RAW_DATA_DIR}")
    logger.info(f"Output directory:   {EXTRACTED_DATA_DIR}")

    results = extract_all_documents()
    # Run the full batch extraction.

    if results:
        logger.info(f"\nExtraction complete. {len(results)} document(s) ready for cleaning.")
        logger.info("Next step: run src/ingestion/cleaner.py")
    else:
        logger.error("Extraction produced no results. Check your PDF files and paths.")
