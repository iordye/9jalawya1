# src/ingestion/chunker.py
# CHUNK STRATEGY — Three-tier Hierarchical Chunking:
#
#   Tier 1: Section-level splits (primary boundary)
#           Detect "Section X." or "S.X" patterns in the text
#           Each section becomes a candidate chunk
#
#   Tier 2: Size-based subdivision (secondary boundary)
#           If a section chunk is too large (>= MAX_CHUNK_CHARS),
#           split it further at paragraph boundaries (\n\n)
#
#   Tier 3: Overlap stitching
#           Each chunk gets a small overlap with the previous chunk
#           so that context is not lost at chunk boundaries

# IMPORTS
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# CONSTANTS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
CHUNKED_DATA_DIR = PROJECT_ROOT / "data" / "chunked"
CHUNKED_DATA_DIR.mkdir(parents=True, exist_ok=True)

#Chunk Size Configuration
MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 100
CHUNK_OVERLAP_CHARS = 150

ALL_CAPS_BLOCKLIST = {
    "ARRANGEMENT OF SECTIONS",
    "FEDERAL REPUBLIC OF NIGERIA",
    "SUBSIDIARY LEGISLATION",
    "NO SUBSIDIARY LEGISLATION",
    "ARRANGEMENT OF RULES",
    "TABLE OF CONTENTS",
}

LEGAL_VERBS = {'shall', 'may', 'must', 'commits', 'liable', 'guilty',
               'entitled', 'prohibited', 'required', 'punishable', 'means'}


# LEGAL SECTION DETECTION PATTERNS
SECTION_PATTERNS = [

    # ── TIER 1: HIGH CONFIDENCE ────────────────────────────────────────────

    # P1: CHAPTER with number/Roman — but only standalone lines
    # WHY BETTER: $ anchor prevents matching "CHAPTER of the agreement" mid-sentence
    # Handles: "CHAPTER C42", "CHAPTER XXVI", "CHAPTER 5"
    # Does NOT match: appendix chapter headers inside tables (handled by H1)
    (r'(?im)^\s*CHAPTER\s+(?:[A-Z]\d+|\d+|[IVXLCDM]{1,8})\s*$', "chapter"),

    # P2: PART with Roman or Arabic — standalone line only
    # WHY BETTER: \s*$ ensures PART is not a mid-sentence reference
    # Handles: "PART I", "PART XIV", "PART 4"
    # Does NOT match: "in respect of PART I of this Act"
    (r'(?im)^\s*PART\s+(?:[IVXLCDM]{1,6}|\d{1,2})\s*$', "part"),

    # P3: "Section X." or "Section X:" — standard keyword form
    # WHY BETTER: allows section numbers like "9A", "106B" (amended sections)
    # Handles: "Section 9.", "Section 9A.", "Section 106."
    (r'(?i)^\s*[Ss]ection\s+(\d{1,4}[A-Z]?)\s*[.:\-]', "section"),

    # P4: "N. Title" — number.space.uppercase or open-paren
    # WHY BETTER: [A-Z\(] catches both "9. Charge of tax" AND "6. (1) Any person"
    # Minimum 10 chars after number prevents matching TOC-style short entries
    # Does NOT match: list items like "1. First item" that are <10 chars total
    (r'(?m)^\s*(\d{1,4}[A-Z]?)\.\s+(?=[A-Z\(]).{9,}', "section"),

    # P5: "N.  Bold Title" — double space variant (Criminal Procedure NS format)
    # WHY BETTER: handles the two-space gap that pdfplumber extracts from
    # bold-numbered sections in two-column PLAC PDFs
    (r'(?m)^\s*(\d{1,4}[A-Z]?)\.\s{2,}(?=[A-Z]).{9,}', "section"),

    # P6: ALL-CAPS standalone heading (2-8 words, no digits)
    # WHY BETTER: requires blank line before it (via lookahead) AND
    # excludes lines with digits (eliminates table chapter headers like
    # "CHAPTER XXVI" which are already caught by P1)
    # Handles: "DUTIES OF SERVICE PROVIDERS", "MISCELLANEOUS"
    # Does NOT match: table chapter headers (caught by H1 table detection)
    (
        r'(?m)(?:^|\n\n)\s*([A-Z][A-Z\s\-]{3,60}[A-Z])\s*(?:\n|$)',
        "section"
    ),

    # P7: SCHEDULE/APPENDIX — standalone line
    # WHY BETTER: optional ordinal prefix handles "FIRST SCHEDULE" through
    # "SIXTH SCHEDULE" plus bare "SCHEDULE" and "APPENDIX"
    (
        r'(?im)^\s*(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH)?\s*'
        r'(?:SCHEDULE|APPENDIX)\s*(?:\[.*?\])?\s*$',
        "schedule"
    ),

    # ── TIER 2: SUBSECTIONS (secondary splits only) ────────────────────────

    # P8: Numbered subsections "(1)", "(2)" — digit AND capital I variant
    # WHY BETTER: handles OCR confusion between "1" and "I" in older PDFs
    # Only fires as secondary split when parent section exceeds MAX_CHUNK_CHARS
    (r'(?m)^\s*\(\s*(?:\d+|[IVX]{1,4})\s*\)\s+[A-Z]', "subsection"),

    # P9: Lettered paragraphs "(a)", "(b)" at line start
    (r'(?m)^\s*\(\s*[a-z]\s*\)\s+', "subsection"),
]

REPEAL_PATTERNS = [
    re.compile(r'\(repealed\s+by', re.IGNORECASE),
    re.compile(r'deleted\s+by\s+(?:No\.|Act)', re.IGNORECASE),
    re.compile(r'omitted\s+by', re.IGNORECASE),
]

# Compile all patterns with appropriate flags
COMPILED_PATTERNS = []
for pattern_str, chunk_type in SECTION_PATTERNS:
    # Extract flags from inline (?im) if present, otherwise use MULTILINE
    if pattern_str.startswith('(?im)'):
        compiled = re.compile(pattern_str[5:], re.IGNORECASE | re.MULTILINE)
    elif pattern_str.startswith('(?i)'):
        compiled = re.compile(pattern_str[4:], re.IGNORECASE)
    elif pattern_str.startswith('(?m)'):
        compiled = re.compile(pattern_str[4:], re.MULTILINE)
    else:
        compiled = re.compile(pattern_str, re.MULTILINE)
    COMPILED_PATTERNS.append((compiled, chunk_type))

def is_false_positive_heading(line: str) -> bool:
    """
    Returns True if an ALL-CAPS matched line is a known false positive
    that should NOT be treated as a section boundary.
 
    Checks:
    1. Exact match against known false-positive phrases
    2. Single word (too short to be a meaningful section heading)
       — "GENERAL", "SECURITY", "COMMITTEE" are sub-heading labels
    3. Continuation phrases ending in prepositions or conjunctions
       — "OR MAKING OF AFFIRMATION", "THAN PRISON OR MENTAL HEALTH ASYLUM"
       indicate these are partial headings extracted mid-section
    """
    stripped = line.strip()
 
    # Check 1: blocklist
    if stripped in ALL_CAPS_BLOCKLIST:
        return True
 
    words = stripped.split()
 
    # Check 2: single word headings — too vague to be a section boundary
    if len(words) == 1:
        return True
 
    # Check 3: starts with a conjunction or preposition — mid-sentence fragment
    FRAGMENT_STARTERS = {'OR', 'AND', 'OF', 'TO', 'THAN', 'FOR', 'IN',
                          'ON', 'AT', 'BY', 'WITH', 'FROM', 'UNDER'}
    if words[0] in FRAGMENT_STARTERS:
        return True
 
    # Check 4: ends with a preposition — dangling phrase
    FRAGMENT_ENDERS = {'OR', 'AND', 'OF', 'TO', 'FOR', 'IN', 'ON',
                       'AT', 'BY', 'WITH', 'FROM', 'THE', 'A', 'AN'}
    if words[-1] in FRAGMENT_ENDERS:
        return True
 
    logger.debug(f"Filtered false heading: {line}")
    return False

def is_table_region(page_text: str) -> bool:
    """
     This function detects tabular regions before chunking 
     and treats them as single atomic chunks regardless of size. 
     This prevents the chunker from splitting tables into meaningless pieces 
     and allows us to flag them with "is_table" metadata for special handling during retrieval.
        Heuristics:
    - Look for lines with "Ditto" or repeated column patterns
    - Check for many lines of similar length (indicating rows)
        Args:
        Page_text (str): The text of the page to analyze
        Returns:
        bool: True if the page is likely a table, False otherwise
    """
    lines = page_text.split('\n')
    # Tables have many short lines with similar lengths
    # and contain "Ditto" or column separator patterns
    ditto_count = sum(1 for l in lines if 'ditto' in l.lower())
    if ditto_count > 3:
        return True
    # Column alignment: many lines of similar length
    line_lengths = [len(l.strip()) for l in lines if l.strip()]
    if line_lengths:
        avg = sum(line_lengths) / len(line_lengths)
        variance = sum((l - avg)**2 for l in line_lengths) / len(line_lengths)
        if variance < 200 and len(line_lengths) > 10:
            return True
    return False

chunk_metadata = {
    # Existing fields — keep all
    "chunk_id": "...",
    "doc_key": "...",
    "document_category": "...",
    "document_year": 0,

    # NEW: Hierarchy path
    "hierarchy_path": "PART II > Section 9",
    # Enables: "Show me all sections under PART II"
    # Enables: LLM citation "Under Part II, Section 9 of the CITA..."
    # Implementation: track current_part and current_chapter during chunking

    # NEW: Section number (parsed, not from heading string)
    "section_number": "9",
    "section_number_int": 9,
    # section_number_int enables range queries: sections 9-15 of CITA
    # Cannot do this with string matching alone

    # NEW: Section title (clean, separated from number)
    "section_title": "Charge of tax",
    # Separate from heading which is "9. Charge of tax"
    # Enables exact title matching and cleaner citation generation

    # NEW: Citation format
    "citation": "Section 9, Companies Income Tax Act (CITA)",
    # Pre-built citation string the LLM can use directly
    # Format: "Section {N}, {full_document_name} ({short_name})"

    # NEW: Is repealed
    "is_repealed": False,
    "repeal_notice": "",
    # Prevents 9jaLawya from citing repealed provisions as current law
    # Critical for CITA which has a repealed Part I

    # NEW: Content type
    "content_type": "provision",
    # Values: "provision", "definition", "penalty", "interpretation",
    #         "schedule", "table", "preamble", "repeal_notice"
    # Enables: filter to only penalty provisions when answering punishment queries

    # NEW: Has penalty
    "has_penalty": True,
    # Boolean flag — True if chunk contains "liable", "imprisonment", "fine"
    # Enables fast pre-filtering: "What is the penalty for X?" → filter has_penalty=True

    # NEW: Parent section for continuations
    "parent_section_heading": "Section 9. Charge of tax",
    "parent_section_number": "9",
    # Already recommended — critical for continuation chunk retrieval

    # NEW: Is table/appendix
    "is_table": False,
    "is_appendix": False,
    # Allows retriever to optionally exclude table chunks from general queries
    # Tables are useful for exact lookups but pollute semantic search
}


def has_legal_content(text: str) -> bool:
    text_lower = text.lower()
    return any(verb in text_lower for verb in LEGAL_VERBS)

# UTILITY FUNCTIONS
def generate_chunk_id(doc_key: str, text: str, index: int) -> str:
    """
    Generates a deterministic unique ID for a chunk based on its content.

    Args:
        doc_key: Document identifier (e.g. "labour_act")
        text: The chunk text content
        index: Position of this chunk within the document

    Returns:
        A 16-character hex string (e.g. "a3f8c2d1e4b7f902")
    """

    # Concatenate the identifying information into one string.
    content_signature = f"{doc_key}_{index}_{text[:200]}"
    return hashlib.md5(content_signature.encode("utf-8")).hexdigest()[:16]


def detect_section_type(line: str) -> Optional[str]:
    """
    Checks if a line matches any of our section boundary patterns.

    Args:
        line: A single line of text

    Returns:
        The chunk_type string ("chapter", "part", "section", "subsection",
        "schedule") if the line is a section boundary, or None if it is not.
    """

    for compiled_pattern, chunk_type in COMPILED_PATTERNS:
        if compiled_pattern.match(line):
           # For ALL-CAPS section matches, apply additional filtering
            # to reject known false positives (document headers,
            # single-word sub-labels, mid-sentence fragments)
           if chunk_type == "section" and line.strip().isupper():
               if is_false_positive_heading(line):
                    continue  # Skip this match, try next pattern
           return chunk_type
    return None

def extract_section_heading(text: str) -> str:
    """
    Extracts the section heading from the first line of a chunk's text.

    Args:
        text: The full text of a chunk

    Returns:
        The heading as a string
    """
    lines = text.strip().split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped:
            # Return the full heading. Slicing at 100 chars damages hybrid search.
            # We cap at 250 just to prevent massive anomalies, but keep words whole.
            if len(stripped) > 250:
                snapped_end = stripped.rfind(' ', 0, 250)
                return stripped[:snapped_end] + "..." if snapped_end != -1 else stripped[:250]
            return stripped
    return "Unknown Section"


def split_by_overlap(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Splits a long text into overlapping sub-chunks at paragraph/sentence boundaries,
    ensuring no mid-word cuts during overlap or hard limits.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    last_start = -1

    while start < len(text):
        if start == last_start:
            logger.warning(f"  Split loop stalled at position {start}, forcing advance.")
            # Force advance to next word boundary to avoid infinite loop
            next_space = text.find(' ', start + 1)
            start = next_space if next_space != -1 else len(text)
            continue
        
        last_start = start
        end = start + max_chars

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # 1. Try paragraph split
        split_point = text.rfind("\n\n", start, end)
        # 2. Try sentence split
        if split_point == -1 or split_point <= start:
            split_point = text.rfind(". ", start, end)
        # 3. Try word split (prevents mid-word cuts on hard limits)
        if split_point == -1 or split_point <= start:
            split_point = text.rfind(" ", start, end)
        # 4. Absolute fallback
        if split_point == -1 or split_point <= start:
            split_point = end

        chunk_text = text[start:split_point].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # OVERLAP LOGIC: Snap to the nearest word boundary after the overlap subtraction
        raw_new_start = split_point - overlap_chars
        if raw_new_start > start:
            # Find the first space AFTER our raw new start to avoid cutting a word
            snapped_start = text.find(' ', raw_new_start)
            if snapped_start != -1 and snapped_start < split_point:
                start = snapped_start + 1 # Start after the space
            else:
                start = split_point # Fallback if no space found
        else:
            start = split_point

    return [c for c in chunks if c]

def extract_section_number(text: str) -> str:
    """
    Extracts the section number from the beginning of a chunk's text.
    Returns empty string if no section number found.
    
    Handles formats:
      "6. (1) Any person..."        → "6"
      "Section 9. Termination..."   → "9"  
      "PART III - OFFENCES..."      → "III"
      "24. Any person who..."       → "24"
    """
    patterns = [
        r'^\s*[Ss]ection\s+(\d+[A-Z]?)\s*[.\-:]',   # "Section 6."
        r'^\s*(\d{1,3}[A-Z]?)\.\s+\(',               # "6. ("
        r'^\s*(\d{1,3}[A-Z]?)\.\s+[A-Z]',            # "24. Any"
        r'^\s*PART\s+([IVXLCDM]+)\s*[-—]',           # "PART III -"
        r'^\s*PART\s+(\d+)\s*[-—]',                   # "PART 4 -"
    ]
    
    first_200 = text[:200]
    for pattern in patterns:
        match = re.search(pattern, first_200)
        if match:
            return match.group(1)
    return ""
    
def validate_section_sequence(sections: list) -> list:
    valid_sections = []
    last_num = 0
    for section in sections:
        num = extract_section_number(section['text'])
        if num and num.isdigit():
            n = int(num)
            if n > last_num * 10:  # jump too large
                section['chunk_type'] = 'table_reference'
                section['is_appendix'] = True
            else:
                last_num = n
        valid_sections.append(section)
    return valid_sections

def extract_section_title(text: str, section_num: str) -> str:
    """Extracts just the title portion without the section number."""
    heading = extract_section_heading(text)
    if section_num and heading.startswith(section_num):
        title = heading[len(section_num):].lstrip('. ').strip()
        return title
    return heading

def check_if_repealed(text: str) -> tuple:
    for pattern in REPEAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, text[match.start():match.start()+80]
    return False, ""

def split_at_subsection_boundaries(text: str, max_chars: int, overlap: int) -> list:
    """
    Splits a long section at subsection markers (1), (2), (3)...
    """
    subsection_pattern = re.compile(
        r'(?m)^\s*\(\s*(?:\d+|[IVX]{1,4})\s*\)\s+[A-Z]'
    )

    split_points = [m.start() for m in subsection_pattern.finditer(text)]

    if not split_points:
        return []

    chunks = []
    current_start = 0

    for split_point in split_points[1:]: 
        candidate = text[current_start:split_point]
        if len(candidate) > max_chars:
            return []
        
        if len(candidate) >= max_chars * 0.5:
            chunks.append(candidate.strip())
            
            # Snap overlap to word boundary
            raw_start = max(0, split_point - overlap)
            if raw_start > 0:
                snapped_start = text.find(' ', raw_start)
                current_start = (snapped_start + 1) if (snapped_start != -1 and snapped_start < split_point) else split_point
            else:
                current_start = 0

    remainder = text[current_start:].strip()
    if remainder:
        chunks.append(remainder)

    return [c for c in chunks if c]

# CORE CHUNKING FUNCTION
def chunk_document_pages(pages, doc_key, metadata):
    """
    4-stage pipeline:
    Stage 0: Pre-classify pages (table vs text)
    Stage 1: Build hierarchy tree (chapter/part/section detection)  
    Stage 2: Size management with semantic split points
    Stage 3: Metadata enrichment
    """

    # ── STAGE 0: PRE-CLASSIFICATION ─────────────────────────────────────────
    classified_pages = []
    for page in pages:
        if is_table_region(page['text']):
            classified_pages.append({**page, 'page_type': 'table'})
        else:
            classified_pages.append({**page, 'page_type': 'text'})

    # Separate table pages — handle them atomically
    table_chunks = []
    text_pages = []
    for page in classified_pages:
        if page['page_type'] == 'table':
            table_chunks.append(page)
        else:
            text_pages.append(page)

    # ── STAGE 1: HIERARCHY DETECTION ────────────────────────────────────────
    full_text = "\n\n".join(p['text'] for p in text_pages if p['text'].strip())

    lines = full_text.split('\n')
    hierarchy_state = {
        'current_chapter': '',
        'current_part': '',
        'current_part_subtitle': '',
    }

    raw_sections = []
    current_lines = []
    current_type = 'preamble'
    current_hierarchy = {}

    for i, line in enumerate(lines):
        section_type = detect_section_type(line)

        if section_type in ('chapter', 'part', 'section', 'schedule'):

            # Update hierarchy state
            if section_type == 'chapter':
                hierarchy_state['current_chapter'] = line.strip()
                hierarchy_state['current_part'] = ''
            elif section_type == 'part':
                hierarchy_state['current_part'] = line.strip()
                # Check next non-empty line for italic subtitle
                for j in range(i+1, min(i+4, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not detect_section_type(next_line):
                        hierarchy_state['current_part_subtitle'] = next_line
                        break

    # Save current accumulation
            if current_lines:
                section_text = '\n'.join(current_lines).strip()
                is_rep, rep_notice = check_if_repealed(section_text)
                if len(section_text) >= MIN_CHUNK_CHARS and has_legal_content(section_text):
                    raw_sections.append({
                        'text': section_text,
                        'type': current_type,
                        'hierarchy': dict(hierarchy_state),
                        'is_repealed': is_rep,
                        'repeal_notice': rep_notice,
                    })
                elif is_rep:
                    # Store repeal notice even if too short
                    raw_sections.append({
                        'text': section_text,
                        'type': 'repeal_notice',
                        'hierarchy': dict(hierarchy_state),
                        'is_repealed': True,
                        'repeal_notice': rep_notice,
                    })

            current_lines = [line]
            current_type = section_type
            current_hierarchy = dict(hierarchy_state)

        else:
            current_lines.append(line)

    # Don't forget last section
    if current_lines:
        section_text = '\n'.join(current_lines).strip()
        if len(section_text) >= MIN_CHUNK_CHARS:
            raw_sections.append({
                'text': section_text,
                'type': current_type,
                'hierarchy': dict(hierarchy_state),
                'is_repealed': False,
                'repeal_notice': '',
            })

    # Section number continuity validation (H2)
    raw_sections = validate_section_sequence(raw_sections)

    # ── STAGE 2: SIZE MANAGEMENT ─────────────────────────────────────────────
    final_chunks = []
    chunk_index = 0
    parent_section_heading = ''

    for section in raw_sections:
        section_text = section['text']
        section_heading = extract_section_heading(section_text)
        section_num = extract_section_number(section_text)
        if not section_num:
            logger.warning(f"Failed to extract section number from: {section_heading}")
            
        section_title = extract_section_title(section_text, section_num)

        # Update parent heading tracking
        if section['type'] == 'section':
            parent_section_heading = section_heading

        if len(section_text) <= MAX_CHUNK_CHARS:
            sub_chunks = [section_text]
        else:
            # Try subsection-boundary split first
            sub_chunks = split_at_subsection_boundaries(
                section_text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS
            )
            if not sub_chunks:
                sub_chunks = split_by_overlap(
                    section_text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS
                )

        for sub_index, chunk_text in enumerate(sub_chunks):
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue

            # CRITICAL: continuation chunks inherit parent heading
            if sub_index == 0:
                heading = section_heading
            else:
                heading = f"{parent_section_heading} [cont. {sub_index}]"

            # Build hierarchy path string
            h = section['hierarchy']
            path_parts = [p for p in [
                h.get('current_chapter', ''),
                h.get('current_part', ''),
                section_heading if sub_index == 0 else parent_section_heading
            ] if p]
            hierarchy_path = ' > '.join(path_parts)

            # Build pre-formatted citation
            # Inside chunk_document_pages_v2, replace the chunk assembly with:
            
            # 1. Build pre-formatted citation
            short_name = metadata.get('short_name', metadata.get('doc_key', ''))
            citation = f"Section {section_num}, {metadata.get('full_name', doc_key)}"
            if short_name:
                citation += f" ({short_name})"

            # 2. CREATE CONTEXTUALIZED TEXT FOR EMBEDDING
            # This dramatically improves vector clustering and semantic retrieval
            contextualized_text = f"[{citation}]\n[Hierarchy: {hierarchy_path}]\n[Title: {section_title}]\n\n{chunk_text}"

            chunk = {
                "chunk_id": generate_chunk_id(doc_key, chunk_text, chunk_index),
                "doc_key": doc_key,
                
                # Use contextualized text for the primary embedding payload
                "text": contextualized_text, 
                # Keep raw text for UI display purposes if needed
                "raw_text": chunk_text,

                "chunk_type": section['type'],
                "heading": heading,
                "section_number": section_num,
                "section_number_int": int(section_num) if section_num.isdigit() else 0,
                "section_title": section_title,
                "hierarchy_path": hierarchy_path,

                "chunk_index": chunk_index,
                "sub_index": sub_index,
                "is_continuation": sub_index > 0,
                "parent_section_heading": parent_section_heading,
                "parent_section_number": extract_section_number(parent_section_heading),

                "is_repealed": section['is_repealed'],
                "repeal_notice": section['repeal_notice'],
                "content_type": classify_content_type(chunk_text),
                "has_penalty": detect_has_penalty(chunk_text),
                "is_table": False,
                "is_appendix": section.get('is_appendix', False),
                "citation": citation,
                
                "full_document_name": metadata.get('full_name', doc_key),
                "document_category": metadata.get('category', 'act'),
                "document_year": metadata.get('year', 0),
                "authority": metadata.get('authority', ''),
                
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
            }

            final_chunks.append(chunk)
            chunk_index += 1

    # Add table chunks at end
    for tc in table_chunks:
        table_text = tc.get('text', '')
        if len(table_text) < MIN_CHUNK_CHARS:
            chunk_index += 1
            continue
 
        table_chunk = {
            # Identity
            "chunk_id": generate_chunk_id(doc_key, table_text, chunk_index),
            "doc_key": doc_key,
 
            # Content
            "text": table_text,
 
            # Structure
            "chunk_type": "table",
            "heading": f"Table (page {tc.get('page_number', '?')})",
            "section_number": "",
            "section_number_int": 0,
            "section_title": "",
            "hierarchy_path": "",
 
            # Continuation — tables are never continuations
            "chunk_index": chunk_index,
            "sub_index": 0,
            "is_continuation": False,
            "parent_section_heading": "",
            "parent_section_number": "",
 
            # Legal state
            "is_repealed": False,
            "repeal_notice": "",
 
            # Content classification
            "content_type": "table",
            "has_penalty": detect_has_penalty(table_text),
            "is_table": True,
            "is_appendix": False,
 
            # Citation
            "citation": f"Table, {metadata.get('full_name', doc_key)}",
 
            # Inherited document metadata
            "full_document_name": metadata.get('full_name', doc_key),
            "document_category": metadata.get('category', 'act'),
            "document_year": metadata.get('year', 0),
            "authority": metadata.get('authority', ''),
 
            # Size
            "char_count": len(table_text),
            "word_count": len(table_text.split()),
        }
 
        final_chunks.append(table_chunk)
        chunk_index += 1
    return final_chunks

def classify_content_type(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ['means ', '"means"', 'shall mean', 'defined as']):
        return 'definition'
    if any(w in text_lower for w in ['liable', 'imprisonment', 'fine of', 'penalty']):
        return 'penalty'
    if 'this act may be cited' in text_lower:
        return 'citation'
    if 'in this act' in text_lower and 'means' in text_lower:
        return 'interpretation'
    return 'provision'


def detect_has_penalty(text: str) -> bool:
    penalty_signals = ['liable', 'imprisonment', 'fine of', 'fine not',
                       'years imprisonment', 'convicted', 'penalty']
    text_lower = text.lower()
    return sum(1 for s in penalty_signals if s in text_lower) >= 2


# DOCUMENT-LEVEL ORCHESTRATOR
def chunk_document(cleaned_data: Dict) -> Optional[Dict]:
    """
    Chunks a single cleaned document and returns a structured result.

    Args:
        cleaned_data: Dictionary from cleaner.py

    Returns:
        Dictionary containing all chunks and updated metadata, or None on failure
    """

    doc_key = cleaned_data.get("doc_key", "unknown")
    pages = cleaned_data.get("pages", [])
    metadata = cleaned_data.get("metadata", {})

    if not pages:
        logger.error(f"No pages found in cleaned data for '{doc_key}'")
        return None

    logger.info(f"Chunking document: {doc_key} ({len(pages)} pages)")
    start_time = time.time()

    chunks = chunk_document_pages(pages, doc_key, metadata)

    elapsed = round(time.time() - start_time, 2)

    if not chunks:
        logger.error(f"No chunks produced for '{doc_key}'")
        return None

    # Calculate chunk size statistics for quality review.
    char_counts = [c["char_count"] for c in chunks]
    avg_chars = round(sum(char_counts) / len(char_counts))
    min_chars = min(char_counts)
    max_chars = max(char_counts)
    continuation_count = sum(1 for c in chunks if c["is_continuation"])

    logger.info(
        f"  Chunks created: {len(chunks)} | "
        f"Avg size: {avg_chars} chars | "
        f"Min: {min_chars} | Max: {max_chars} | "
        f"Continuations: {continuation_count} | "
        f"Time: {elapsed}s"
    )

    # Warn if chunk sizes look problematic.
    if avg_chars < 200:
        logger.warning(
            f"  Average chunk size is very small ({avg_chars} chars). "
            f"Section detection may not be working well for this document. "
            f"Review chunked output."
        )
    if max_chars > MAX_CHUNK_CHARS * 1.5:
        logger.warning(
            f"  Some chunks exceed 1.5x the max size. "
            f"Consider reviewing split logic for this document."
        )

    return {
        "doc_key": doc_key,
        "metadata": {
            **metadata,
            "total_chunks": len(chunks),
            "avg_chunk_chars": avg_chars,
            "min_chunk_chars": min_chars,
            "max_chunk_chars": max_chars,
            "continuation_chunks": continuation_count,
            "chunking_time_seconds": elapsed,
        },
        "chunks": chunks,
    }

# SAVE FUNCTION
def save_chunked_document(chunked_data: Dict, output_dir: Path) -> Path:
    """
    Saves the chunked document to a JSON file.

    Args:
        chunked_data: The dictionary returned by chunk_document()
        output_dir: The folder to save the JSON file into

    Returns:
        Path to the saved JSON file
    """

    output_path = output_dir / f"{chunked_data['doc_key']}_chunked.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=2)

    logger.info(f"  Saved to: {output_path.name}")
    return output_path


# BATCH PROCESSING FUNCTION
def chunk_all_documents(
    cleaned_dir: Path = CLEANED_DATA_DIR,
    output_dir: Path = CHUNKED_DATA_DIR,
) -> List[Dict]:
    """
    Chunks ALL cleaned JSON files in the cleaned data directory.

    Args:
        cleaned_dir: Folder containing cleaned JSON files
        output_dir: Folder to save chunked JSON files

    Returns:
        List of all successfully chunked document dictionaries
    """

    if not cleaned_dir.exists():
        logger.error(f"Cleaned data directory not found: {cleaned_dir}")
        logger.error("Please run cleaner.py first.")
        return []

    cleaned_files = list(cleaned_dir.glob("*_cleaned.json"))

    if not cleaned_files:
        logger.error(f"No cleaned JSON files found in {cleaned_dir}")
        logger.error("Please run cleaner.py first.")
        return []

    logger.info(f"Found {len(cleaned_files)} cleaned document(s) to chunk:")
    for f in cleaned_files:
        logger.info(f"  - {f.name}")

    all_chunked = []

    for cleaned_path in sorted(cleaned_files):

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {cleaned_path.name}")
        logger.info(f"{'='*60}")

        try:
            with open(cleaned_path, "r", encoding="utf-8") as f:
                cleaned_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {cleaned_path.name}: {e}")
            continue

        chunked_data = chunk_document(cleaned_data)

        if chunked_data is None:
            logger.error(f"Chunking failed for {cleaned_path.name}")
            continue

        save_chunked_document(chunked_data, output_dir)
        all_chunked.append(chunked_data)

    # Final summary with per-document chunk counts.
    logger.info(f"\n{'='*60}")
    logger.info(f"CHUNKING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Documents processed:    {len(cleaned_files)}")
    logger.info(f"Successfully chunked:   {len(all_chunked)}")
    logger.info(f"Failed:                 {len(cleaned_files) - len(all_chunked)}")

    total_chunks = sum(d["metadata"]["total_chunks"] for d in all_chunked)
    logger.info(f"Total chunks produced:  {total_chunks}")
    logger.info(f"\nPer-document breakdown:")

    for doc in all_chunked:
        meta = doc["metadata"]
        logger.info(
            f"  {doc['doc_key']:<35} "
            f"{meta['total_chunks']:>4} chunks | "
            f"avg {meta['avg_chunk_chars']} chars"
        )
        # :<35 left-aligns in 35-char field (document name)
        # :>4 right-aligns in 4-char field (chunk count)

    return all_chunked


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("9jaLawya - Chunking Stage")
    logger.info(f"Input directory:  {CLEANED_DATA_DIR}")
    logger.info(f"Output directory: {CHUNKED_DATA_DIR}")

    results = chunk_all_documents()

    if results:
        total = sum(d["metadata"]["total_chunks"] for d in results)
        logger.info(
            f"\nChunking complete. {len(results)} document(s) → {total} total chunks."
        )
        logger.info("Next step: run src/ingestion/ingestor.py")
    else:
        logger.error("Chunking produced no results. Check cleaned files and paths.")
