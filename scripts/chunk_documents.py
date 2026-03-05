"""
Chunk extracted text files by Dieu (Article) and Khoan (Clause).
Each chunk preserves metadata: document name, chapter, article number, clause number.
Saves chunks as JSON in data/chunks/
"""
import json
import os
import re
import sys
import unicodedata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "data", "chunks")

# Regex patterns for Vietnamese legal document structure
# Matches: "Điều 1.", "Điều 12.", "Điều 123." etc. (with optional OCR noise prefix)
DIEU_PATTERN = re.compile(
    r"^[^\w]*?(Điều\s+\d+)\.\s*(.*)$", re.MULTILINE
)

# Matches: "Chương I", "Chương II", "CHƯƠNG I", OCR variants
CHUONG_PATTERN = re.compile(
    r"^(?:CHƯƠNG|Chương|CHUONG|Chuong)\s+([IVXLCDM]+|\d+)\s*$", re.MULTILINE | re.IGNORECASE
)

# Matches clause numbers: "1.", "2.", "3." at the start of a line (Khoan)
KHOAN_PATTERN = re.compile(r"^(\d+)\.\s+", re.MULTILINE)

# Matches point labels: "a)", "b)", "c)" at the start of a line (Diem)
DIEM_PATTERN = re.compile(r"^([a-zđ])\)\s+", re.MULTILINE)

# Document name mapping for cleaner metadata
DOC_NAME_MAP = {
    "LUẬT ĐẤT ĐAI 2024": "Luat Dat Dai 2024 (Luat so 31/2024/QH15)",
    "102-cp.signed": "Nghi dinh 102/2024/ND-CP (Quy dinh chi tiet thi hanh Luat Dat dai)",
    "103. 2024 quy định về tiền sử dụng đất, tiền thuê đất": "Nghi dinh 103/2024/ND-CP (Tien su dung dat, tien thue dat)",
    "151-ndcp.signed": "Nghi dinh 151/2025/ND-CP (Phan dinh tham quyen chinh quyen dia phuong 02 cap)",
    "226nd.signed": "Nghi dinh 226/2025/ND-CP (Sua doi bo sung cac nghi dinh thi hanh Luat Dat dai)",
    "254. NQ TW": "Nghi quyet 254/2025/QH15 (Thao go vuong mac thi hanh Luat Dat dai)",
    "49-2026ndcp.signed": "Nghi dinh 49/2026/ND-CP (Sua doi bo sung nghi dinh chi tiet Luat Dat dai)",
    "71-cp.signed": "Nghi dinh 71/2024/ND-CP (Quy dinh ve gia dat)",
    "88nd.signed": "Nghi dinh 88/2024/ND-CP (Boi thuong, ho tro, tai dinh cu khi thu hoi dat)",
}


def clean_text(text: str) -> str:
    """Remove page markers, OCR artifacts, and clean up whitespace."""
    # Remove page markers from extraction
    text = re.sub(r"--- Trang \d+ ---\n?", "", text)
    # Remove digital signature metadata blocks
    text = re.sub(r"Người ký:.*?(?:\n.*?){0,5}Thời gian ký:.*?\n", "", text, flags=re.DOTALL)
    # Remove common OCR noise characters
    text = re.sub(r"[|]{2,}", "", text)
    text = re.sub(r"\.{4,}", "", text)
    # Remove stray pipe, backtick, underscore at end of lines (OCR artifacts)
    text = re.sub(r"\s*[|`_¬ˆ]+\s*$", "", text, flags=re.MULTILINE)
    # Remove lines that are just OCR garbage (short non-Vietnamese fragments)
    text = re.sub(r"^[|`_¬ˆ\s\-\.]{1,5}$", "", text, flags=re.MULTILINE)
    # Fix OCR "ó." -> "6." pattern (common misread)
    text = re.sub(r"^ó\.", "6.", text, flags=re.MULTILINE)
    text = re.sub(r"^§\.", "8.", text, flags=re.MULTILINE)
    # Fix common OCR misreads for Vietnamese diacritics
    text = re.sub(r"(?<!\w)Diéu(?!\w)", "Điều", text)
    text = re.sub(r"(?<!\w)Dieu(?!\w)", "Điều", text)
    # Normalize whitespace but keep paragraph structure
    text = re.sub(r" {3,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just numbers (page numbers from OCR)
    text = re.sub(r"^\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def get_doc_name(filename: str) -> str:
    """Get a clean document name from filename."""
    base = os.path.splitext(filename)[0]
    # Normalize Unicode (macOS uses NFD, our keys are NFC)
    base_nfc = unicodedata.normalize("NFC", base)
    return DOC_NAME_MAP.get(base_nfc, DOC_NAME_MAP.get(base, base_nfc))


def find_chapters(text: str) -> list[dict]:
    """Find all chapter positions and titles."""
    chapters = []
    for match in CHUONG_PATTERN.finditer(text):
        # Try to get chapter title from the next line
        end_pos = match.end()
        next_newline = text.find("\n", end_pos + 1)
        if next_newline > 0:
            title_candidate = text[end_pos:next_newline].strip()
        else:
            title_candidate = ""

        chapters.append({
            "number": match.group(1),
            "title": title_candidate,
            "start": match.start(),
        })
    return chapters


def get_chapter_for_position(chapters: list[dict], pos: int) -> str:
    """Find which chapter a position belongs to."""
    current_chapter = ""
    for ch in chapters:
        if ch["start"] <= pos:
            current_chapter = f"Chuong {ch['number']}: {ch['title']}"
        else:
            break
    return current_chapter


def chunk_by_dieu(text: str, doc_name: str) -> list[dict]:
    """Split text into chunks by Dieu (Article).
    Each Dieu becomes one chunk with its full content.
    If a Dieu is very long, further split by Khoan (Clause).
    """
    chapters = find_chapters(text)
    chunks = []

    # Find all article positions
    dieu_matches = list(DIEU_PATTERN.finditer(text))

    if not dieu_matches:
        # No articles found - treat the whole document as one chunk
        chunks.append({
            "doc_name": doc_name,
            "chapter": "",
            "dieu": "",
            "content": text[:3000] if len(text) > 3000 else text,
            "metadata_str": doc_name,
        })
        return chunks

    for i, match in enumerate(dieu_matches):
        dieu_label = match.group(1)  # e.g., "Điều 1"
        dieu_title = match.group(2).strip()  # Article title

        # Get content until next article or end
        start = match.start()
        if i + 1 < len(dieu_matches):
            end = dieu_matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()
        chapter = get_chapter_for_position(chapters, start)

        # If content is too long (>2000 chars), split by Khoan
        if len(content) > 2000:
            sub_chunks = split_by_khoan(content, doc_name, chapter, dieu_label, dieu_title)
            if sub_chunks:
                chunks.extend(sub_chunks)
                continue

        chunks.append({
            "doc_name": doc_name,
            "chapter": chapter,
            "dieu": dieu_label,
            "dieu_title": dieu_title,
            "content": content,
            "metadata_str": f"{doc_name} | {chapter} | {dieu_label}. {dieu_title}".strip(" |"),
        })

    return chunks


def split_by_khoan(
    content: str, doc_name: str, chapter: str, dieu_label: str, dieu_title: str
) -> list[dict]:
    """Split a long Dieu into smaller chunks by Khoan (Clause)."""
    chunks = []

    # Find all Khoan positions
    khoan_matches = list(KHOAN_PATTERN.finditer(content))

    if len(khoan_matches) < 2:
        # Not enough clauses to split meaningfully
        # Just split into ~1500 char chunks with overlap
        return split_by_size(content, doc_name, chapter, dieu_label, dieu_title)

    # Include the header (before first Khoan) with each chunk for context
    header = content[: khoan_matches[0].start()].strip()

    for i, match in enumerate(khoan_matches):
        khoan_num = match.group(1)

        start = match.start()
        if i + 1 < len(khoan_matches):
            end = khoan_matches[i + 1].start()
        else:
            end = len(content)

        khoan_content = content[start:end].strip()

        # Include header for context
        full_content = f"{header}\n\n{khoan_content}" if header else khoan_content

        # If still too long, split by size
        if len(full_content) > 2500:
            sub_chunks = split_by_size(
                full_content, doc_name, chapter, dieu_label, dieu_title, f"Khoan {khoan_num}"
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append({
                "doc_name": doc_name,
                "chapter": chapter,
                "dieu": dieu_label,
                "dieu_title": dieu_title,
                "khoan": khoan_num,
                "content": full_content,
                "metadata_str": (
                    f"{doc_name} | {chapter} | {dieu_label}. {dieu_title} | Khoan {khoan_num}"
                ).strip(" |"),
            })

    return chunks


def split_by_size(
    content: str,
    doc_name: str,
    chapter: str,
    dieu_label: str,
    dieu_title: str,
    khoan: str = "",
    max_size: int = 1500,
    overlap: int = 200,
) -> list[dict]:
    """Split content into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    part = 1

    while start < len(content):
        end = start + max_size
        # Try to break at a sentence or paragraph boundary
        if end < len(content):
            # Look for a good break point
            for sep in ["\n\n", "\n", ". ", "; "]:
                last_sep = content.rfind(sep, start + max_size // 2, end)
                if last_sep > 0:
                    end = last_sep + len(sep)
                    break

        chunk_text = content[start:end].strip()
        if chunk_text:
            label = f"{dieu_label}"
            if khoan:
                label += f" {khoan}"
            label += f" (phan {part})"

            chunks.append({
                "doc_name": doc_name,
                "chapter": chapter,
                "dieu": dieu_label,
                "dieu_title": dieu_title,
                "content": chunk_text,
                "metadata_str": f"{doc_name} | {chapter} | {label}".strip(" |"),
            })
            part += 1

        start = end - overlap if end < len(content) else end

    return chunks


def process_file(filepath: str) -> list[dict]:
    """Process a single text file into chunks."""
    filename = os.path.basename(filepath)
    doc_name = get_doc_name(filename)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_text(text)
    chunks = chunk_by_dieu(text, doc_name)

    return chunks


def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt"))
    if not txt_files:
        print(f"Khong tim thay file .txt nao trong {RAW_DIR}")
        print("Hay chay extract_pdf.py truoc.")
        sys.exit(1)

    all_chunks = []
    print(f"Xu ly {len(txt_files)} file:")

    for txt_file in txt_files:
        filepath = os.path.join(RAW_DIR, txt_file)
        chunks = process_file(filepath)
        all_chunks.extend(chunks)
        print(f"  - {txt_file}: {len(chunks)} chunks")

    # Save all chunks to a single JSON file
    output_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Also save per-document chunk files for reference
    by_doc = {}
    for chunk in all_chunks:
        doc = chunk["doc_name"]
        by_doc.setdefault(doc, []).append(chunk)

    for doc_name, doc_chunks in by_doc.items():
        safe_name = re.sub(r"[^\w\s-]", "", doc_name).replace(" ", "_")
        doc_path = os.path.join(CHUNKS_DIR, f"{safe_name}.json")
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(doc_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTong cong: {len(all_chunks)} chunks")
    print(f"Da luu vao: {output_path}")

    # Print summary
    print("\nThong ke theo van ban:")
    for doc_name, doc_chunks in sorted(by_doc.items()):
        print(f"  {doc_name}: {len(doc_chunks)} chunks")


if __name__ == "__main__":
    main()
