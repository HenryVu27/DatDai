"""
Chunk markdown (.md) files by Dieu (Article).
Handles inconsistent heading levels across documents:
  - #{1,6} Dieu N. Title
  - **Dieu N. Title**
Splits long Dieu by Khoan (Clause) boundaries.
Saves chunks as JSON in data/chunks/all_chunks.json
"""
import json
import os
import re
import sys
import unicodedata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "data", "chunks")

DOC_NAME_MAP = {
    "LUẬT ĐẤT ĐAI 2024": "Luat Dat Dai 2024 (31/2024/QH15)",
    "102-cp.signed": "Nghi dinh 102/2024/ND-CP",
    "103. 2024 quy định về tiền sử dụng đất, tiền thuê đất": "Nghi dinh 103/2024/ND-CP",
    "151-ndcp.signed": "Nghi dinh 151/2025/ND-CP",
    "226nd.signed": "Nghi dinh 226/2025/ND-CP",
    "254. NQ TW": "Nghi quyet 254/2025/QH15",
    "49-2026ndcp.signed": "Nghi dinh 49/2026/ND-CP",
    "71-cp.signed": "Nghi dinh 71/2024/ND-CP",
    "88nd.signed": "Nghi dinh 88/2024/ND-CP",
    "12-2024ndcp": "Nghi dinh 12/2024/ND-CP",
    "50-2026ndcp": "Nghi dinh 50/2026/ND-CP",
}

# Reverse mapping: doc_name -> doc_id (short form used in cross-references)
DOC_ID_MAP = {
    "Luat Dat Dai 2024 (31/2024/QH15)": "ldd2024",
    "Nghi dinh 71/2024/ND-CP": "nd71",
    "Nghi dinh 88/2024/ND-CP": "nd88",
    "Nghi dinh 102/2024/ND-CP": "nd102",
    "Nghi dinh 103/2024/ND-CP": "nd103",
    "Nghi dinh 151/2025/ND-CP": "nd151",
    "Nghi dinh 226/2025/ND-CP": "nd226",
    "Nghi quyet 254/2025/QH15": "nq254",
    "Nghi dinh 49/2026/ND-CP": "nd49",
    "Nghi dinh 12/2024/ND-CP": "nd12",
    "Nghi dinh 50/2026/ND-CP": "nd50",
}

# Amendment decree doc_ids
AMENDMENT_DOC_IDS = {"nd226", "nd49", "nd50"}

# Regex for Dieu heading: markdown headings or bold
# Must be at line start, not mid-sentence
# Matches:
#   ### Điều N. Title
#   **Điều N. Title**
#   **Điều N.** Title
DIEU_HEADING_RE = re.compile(
    r'^(?:#{1,6}\s+|\*\*|["\u201c\u201d])(Điều\s+\d+[a-zđ]?)\.\s*(?:\*\*\s*)?(.+?)(?:\*\*|["\u201c\u201d])?$'
)

# Regex for Chuong heading: markdown headings only
CHUONG_HEADING_RE = re.compile(
    r"^#{1,6}\s+Chương\s+([IVXLCDM]+)",
    re.IGNORECASE,
)

# Khoan pattern: "1. ", "2. " etc at line start
KHOAN_RE = re.compile(r"^(\d+)\.\s+", re.MULTILINE)

# Digital signature block patterns
SIGNATURE_PATTERNS = [
    re.compile(r"^VGP CỔNG THÔNG TIN ĐIỆN TỬ CHÍNH PHỦ$"),
    re.compile(r"^Người ký:.*$"),
    re.compile(r"^Email:.*$"),
    re.compile(r"^Cơ quan:.*$"),
    re.compile(r"^Thời gian ký:.*$"),
]


def parse_dieu_heading(line: str):
    """Parse a Dieu heading line.

    Returns ("Dieu N", "Title") or None.
    Only matches when Dieu appears at the start of a heading, not mid-sentence.
    """
    line = line.strip()
    m = DIEU_HEADING_RE.match(line)
    if m:
        return (m.group(1), m.group(2).strip())
    return None


def find_chapters(text: str):
    """Find all chapter headings in text.

    Returns list of {"number": "I", "title": "...", "line_idx": N}
    Title is the next non-empty line after the chapter heading line.
    """
    lines = text.split("\n")
    chapters = []
    for i, line in enumerate(lines):
        m = CHUONG_HEADING_RE.match(line.strip())
        if m:
            number = m.group(1).upper()
            # Find title: next non-empty line (strip markdown heading markers)
            title = ""
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j].strip()
                if candidate:
                    # Remove markdown heading markers and bold
                    title = re.sub(r"^#{1,6}\s+", "", candidate)
                    title = title.strip("*").strip()
                    break
            chapters.append({
                "number": number,
                "title": title,
                "line_idx": i,
            })
    return chapters


def find_all_dieu(text: str, doc_name: str):
    """Find all Dieu in text and extract their content.

    Returns list of chunk dicts with keys:
      doc_name, chapter, dieu, dieu_title, content, metadata_str
    """
    lines = text.split("\n")

    # Build chapter map: line_idx -> chapter info
    chapters = find_chapters(text)

    def get_chapter_at(line_idx):
        """Get the chapter that contains the given line index."""
        current_chapter = ""
        for ch in chapters:
            if ch["line_idx"] <= line_idx:
                current_chapter = f"Chương {ch['number']}: {ch['title']}"
            else:
                break
        return current_chapter

    # Find all Dieu positions
    dieu_positions = []
    for i, line in enumerate(lines):
        parsed = parse_dieu_heading(line)
        if parsed:
            dieu_positions.append({
                "line_idx": i,
                "dieu": parsed[0],
                "dieu_title": parsed[1],
            })

    if not dieu_positions:
        # No Dieu found - return single chunk with truncated content
        content = text.strip()
        if len(content) > 3000:
            content = content[:3000] + "..."
        return [{
            "doc_name": doc_name,
            "chapter": "",
            "dieu": "",
            "dieu_title": "",
            "content": content,
            "metadata_str": f"[{doc_name}]",
        }]

    chunks = []
    for idx, pos in enumerate(dieu_positions):
        start = pos["line_idx"]
        if idx + 1 < len(dieu_positions):
            end = dieu_positions[idx + 1]["line_idx"]
        else:
            end = len(lines)

        content = "\n".join(lines[start:end]).strip()
        chapter = get_chapter_at(start)

        metadata_str = f"[{doc_name}]"
        if chapter:
            metadata_str += f" [{chapter}]"
        metadata_str += f" [{pos['dieu']}. {pos['dieu_title']}]"

        chunks.append({
            "doc_name": doc_name,
            "chapter": chapter,
            "dieu": pos["dieu"],
            "dieu_title": pos["dieu_title"],
            "content": content,
            "metadata_str": metadata_str,
        })

    return chunks


def split_long_dieu(chunk: dict, max_size: int = 2000):
    """Split a long Dieu chunk by Khoan boundaries.

    If content <= max_size chars, return [chunk].
    Try splitting by Khoan (numbered clauses).
    Fallback: fixed-size split with 200 char overlap at sentence boundaries.
    """
    content = chunk["content"]
    if len(content) <= max_size:
        return [chunk]

    lines = content.split("\n")

    # Find Khoan boundaries (skip the first line which is the Dieu heading)
    khoan_positions = []  # (line_index, khoan_number)
    for i, line in enumerate(lines):
        if i == 0:
            continue
        m = re.match(r"^(\d+)\.\s+", line)
        if m:
            khoan_positions.append((i, int(m.group(1))))

    if len(khoan_positions) >= 2:
        # Split at Khoan boundaries
        # Build the Dieu header (first line(s) before first Khoan)
        header_end = khoan_positions[0][0]
        header = "\n".join(lines[:header_end]).strip()

        sub_chunks = []
        for k_idx, (line_idx, khoan_num) in enumerate(khoan_positions):
            if k_idx + 1 < len(khoan_positions):
                next_line_idx = khoan_positions[k_idx + 1][0]
            else:
                next_line_idx = len(lines)

            khoan_content = "\n".join(lines[line_idx:next_line_idx]).strip()
            sub_content = header + "\n\n" + khoan_content

            sub_chunk = dict(chunk)
            sub_chunk["content"] = sub_content
            sub_chunk["metadata_str"] = chunk["metadata_str"] + f" [Khoản {khoan_num}]"
            sub_chunks.append(sub_chunk)

        # Recursively split any still-oversized Khoan chunks by size
        result = []
        for sc in sub_chunks:
            if len(sc["content"]) > max_size:
                result.extend(_split_by_size(sc, max_size, overlap=200))
            else:
                result.append(sc)
        return result

    # Fallback: fixed-size split with overlap
    return _split_by_size(chunk, max_size, overlap=200)


def _split_by_size(chunk: dict, max_size: int, overlap: int = 200):
    """Split chunk content into fixed-size pieces with overlap at sentence boundaries."""
    content = chunk["content"]
    pieces = []
    start = 0
    part_num = 0

    while start < len(content):
        end = start + max_size

        if end >= len(content):
            piece = content[start:]
        else:
            # Try to break at a sentence boundary (. or \n)
            # Search backwards from end for a good break point
            break_point = end
            search_region = content[max(start, end - 300):end]
            # Find last newline followed by a blank line or period
            last_newline = search_region.rfind("\n")
            if last_newline > 0:
                break_point = max(start, end - 300) + last_newline + 1
            else:
                last_period = search_region.rfind(".")
                if last_period > 0:
                    break_point = max(start, end - 300) + last_period + 1

            piece = content[start:break_point]
            # Next start with overlap
            end = break_point

        part_num += 1
        sub_chunk = dict(chunk)
        sub_chunk["content"] = piece.strip()
        sub_chunk["metadata_str"] = chunk["metadata_str"] + f" [Phần {part_num}]"
        pieces.append(sub_chunk)

        if end >= len(content):
            break

        start = max(0, end - overlap)

    return pieces


def clean_md_text(text: str) -> str:
    """Clean markdown text:
    - Remove digital signature blocks at top
    - Remove markdown table separator rows
    - Remove standalone page numbers
    - Collapse 3+ blank lines to 2
    """
    lines = text.split("\n")

    # Remove digital signature block at top (first N lines)
    # Look for signature patterns in the first 10 lines
    sig_end = 0
    for i, line in enumerate(lines[:10]):
        stripped = line.strip()
        for pat in SIGNATURE_PATTERNS:
            if pat.match(stripped):
                sig_end = i + 1
                break

    if sig_end > 0:
        # Remove all lines up to and including the last signature line
        # Also remove any blank lines immediately after
        while sig_end < len(lines) and not lines[sig_end].strip():
            sig_end += 1
        lines = lines[sig_end:]

    cleaned = []
    for line in lines:
        stripped = line.strip()

        # Remove table separator rows like "| --- | --- |"
        if re.match(r"^\|[\s\-:|]+\|$", stripped):
            continue

        # Remove standalone page numbers (just a number on its own line)
        if re.match(r"^\d{1,4}$", stripped):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text


# ---------- Amendment index builder ----------

# Regex to detect target ND number from a Dieu title in amendment decrees.
# Matches patterns like "Nghị định số 71/2024/NĐ-CP" or "Nghị định số 103/2024/NĐ-CP"
_TARGET_ND_RE = re.compile(
    r"Nghị định số (\d+)/\d{4}/NĐ-CP",
)

# Map ND number -> doc_id for target resolution
_ND_NUM_TO_DOC_ID = {
    "71": "nd71", "88": "nd88", "101": "nd101", "102": "nd102",
    "102": "nd102", "103": "nd103", "151": "nd151", "226": "nd226",
    "49": "nd49", "50": "nd50", "12": "nd12", "291": "nd103",
}

# Regex patterns for amendment types within chunk content
_AMEND_PATTERNS = [
    # "Sua doi, bo sung ... Dieu X" or "Sua doi ... khoan Y Dieu X"
    (re.compile(
        r"Sửa đổi,?\s*bổ sung\s+.*?(?:khoản\s+\d+\s+)?Điều\s+(\d+[a-zđ]?)",
        re.IGNORECASE,
    ), "sua_doi"),
    # "Thay the ... Dieu X"
    (re.compile(
        r"Thay thế\s+.*?Điều\s+(\d+[a-zđ]?)",
        re.IGNORECASE,
    ), "thay_the"),
    # "Bo sung Dieu Xa" (adding new article)
    (re.compile(
        r"Bổ sung\s+Điều\s+(\d+[a-zđ]+)",
        re.IGNORECASE,
    ), "bo_sung"),
    # "Bai bo ... Dieu X"
    (re.compile(
        r"Bãi bỏ\s+.*?Điều\s+(\d+[a-zđ]?)",
        re.IGNORECASE,
    ), "bai_bo"),
]


def _resolve_target_doc(dieu_title: str) -> str | None:
    """Extract target doc_id from a Dieu title of an amendment decree.

    E.g. "Sua doi, bo sung mot so dieu cua Nghi dinh so 71/2024/ND-CP ..."
    -> "nd71"
    """
    m = _TARGET_ND_RE.search(dieu_title)
    if m:
        nd_num = m.group(1)
        return _ND_NUM_TO_DOC_ID.get(nd_num)
    return None


def _detect_amendments_in_content(content: str) -> list[dict]:
    """Detect amendment targets (Dieu numbers + type) within chunk content.

    Returns list of {"target_dieu": "Dieu X", "type": "sua_doi"|...}.
    """
    results = []
    seen = set()
    for pattern, atype in _AMEND_PATTERNS:
        for m in pattern.finditer(content):
            dieu_num = m.group(1)
            key = (f"Điều {dieu_num}", atype)
            if key not in seen:
                seen.add(key)
                results.append({
                    "target_dieu": f"Điều {dieu_num}",
                    "type": atype,
                })
    return results


def build_amendment_index(all_chunks: list[dict]) -> dict:
    """Build amendment index mapping from amendment decrees to their targets.

    Scans chunks from amendment decrees (nd226, nd49, nd50), detects
    which Dieu in which target decree each chunk amends, and saves
    the result to data/amendment_index.json.

    Returns the index dict.
    """
    index = {}

    # Group amendment chunks by (source_doc_id, source_dieu)
    # so we can resolve the target doc from the Dieu title once per Dieu.
    source_groups: dict[tuple[str, str], list[dict]] = {}

    for chunk in all_chunks:
        doc_name = chunk.get("doc_name", "")
        doc_id = DOC_ID_MAP.get(doc_name)
        if not doc_id or doc_id not in AMENDMENT_DOC_IDS:
            continue
        dieu = chunk.get("dieu", "")
        if not dieu:
            continue
        key = (doc_id, dieu)
        source_groups.setdefault(key, []).append(chunk)

    for (source_doc_id, source_dieu), chunks in source_groups.items():
        # Resolve target doc from the Dieu title (all chunks from same Dieu
        # share the same dieu_title)
        dieu_title = chunks[0].get("dieu_title", "")
        target_doc_id = _resolve_target_doc(dieu_title)
        if not target_doc_id:
            continue

        index_key = f"{source_doc_id}->{target_doc_id}"
        if index_key not in index:
            index[index_key] = {
                "source_doc": source_doc_id,
                "target_doc": target_doc_id,
                "amendments": [],
            }

        # Scan each chunk for specific amendment targets
        for chunk in chunks:
            content = chunk.get("content", "")
            amendments = _detect_amendments_in_content(content)
            chunk_id = chunk.get("chunk_id", "")

            for amend in amendments:
                # Check if we already have this exact amendment recorded
                existing = None
                for a in index[index_key]["amendments"]:
                    if (a["source_dieu"] == source_dieu
                            and a["target_dieu"] == amend["target_dieu"]
                            and a["type"] == amend["type"]):
                        existing = a
                        break

                if existing:
                    if chunk_id not in existing["chunk_ids"]:
                        existing["chunk_ids"].append(chunk_id)
                else:
                    index[index_key]["amendments"].append({
                        "source_dieu": source_dieu,
                        "target_dieu": amend["target_dieu"],
                        "type": amend["type"],
                        "chunk_ids": [chunk_id],
                    })

    # Save to JSON
    output_path = os.path.join(DATA_DIR, "amendment_index.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    # Print stats
    total_amendments = sum(
        len(v["amendments"]) for v in index.values()
    )
    print(f"\n  Amendment index: {len(index)} relationships, "
          f"{total_amendments} amendment entries")
    print(f"  Saved to: {output_path}")

    return index


def process_file(filepath: str):
    """Read .md file, clean, find all Dieu, split long ones.

    Returns list of chunk dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_md_text(text)

    # Determine doc_name from filename (NFC normalize for macOS NFD filenames)
    basename = unicodedata.normalize("NFC", os.path.splitext(os.path.basename(filepath))[0])
    doc_name = DOC_NAME_MAP.get(basename, basename)

    chunks = find_all_dieu(text, doc_name)

    # Split long chunks
    result = []
    for chunk in chunks:
        result.extend(split_long_dieu(chunk))

    return result


def main():
    """Process all .md files from data/, assign chunk_ids, save to all_chunks.json."""
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    md_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".md")
    )

    if not md_files:
        print("No .md files found in", DATA_DIR)
        sys.exit(1)

    all_chunks = []
    total_by_doc = {}

    for md_file in md_files:
        filepath = os.path.join(DATA_DIR, md_file)
        chunks = process_file(filepath)
        basename = unicodedata.normalize("NFC", os.path.splitext(md_file)[0])
        doc_name = DOC_NAME_MAP.get(basename, basename)
        total_by_doc[doc_name] = len(chunks)
        all_chunks.extend(chunks)

    # Assign chunk_ids
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"chunk_{i:04d}"

    # Save
    output_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Build amendment index
    build_amendment_index(all_chunks)

    # Print stats
    print(f"\n{'='*60}")
    print(f"Chunking complete")
    print(f"{'='*60}")
    for doc_name, count in total_by_doc.items():
        print(f"  {doc_name}: {count} chunks")
    print(f"{'='*60}")
    print(f"  TOTAL: {len(all_chunks)} chunks")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    # Print size stats
    sizes = [len(c["content"]) for c in all_chunks]
    print(f"\n  Content size stats:")
    print(f"    Min: {min(sizes)} chars")
    print(f"    Max: {max(sizes)} chars")
    print(f"    Avg: {sum(sizes) // len(sizes)} chars")
    over_2k = sum(1 for s in sizes if s > 2000)
    print(f"    Over 2000 chars: {over_2k} chunks")


if __name__ == "__main__":
    main()
