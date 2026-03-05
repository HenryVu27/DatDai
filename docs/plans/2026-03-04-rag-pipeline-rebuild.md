# RAG Pipeline Rebuild Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the entire RAG pipeline from chunking .md files through Qdrant indexing to a production-grade hybrid retriever with cross-encoder reranking, payload filtering, HyDE, relevance threshold, and keyword fallback.

**Architecture:** Structure-aware markdown chunker reads 9 .md legal documents, splits by Dieu/Khoan with parent-child references, stores in Qdrant Cloud with dense (Gemini) + sparse (TF-IDF) vectors. Retriever uses hybrid search with RRF fusion, cross-encoder reranking (multilingual bge-reranker-v2-m3), metadata filtering, HyDE for legal language bridging, relevance threshold, and keyword fallback. Modular `app/rag/` package replaces monolithic `app/rag.py`.

**Tech Stack:** Python 3.12, Qdrant Cloud (existing), Gemini embedding-001 (existing), fastembed (cross-encoder ONNX), FastAPI (existing)

---

## Current State

- **Source data:** 9 `.md` files in `data/` (~2.9MB total, ~24K lines, ~543 Dieu)
- **Heading inconsistency:** Dieu appears as `###`, `##`, `#`, `####`, and `**bold**` across documents
- **Existing chunker:** `scripts/chunk_documents.py` reads `.txt` from `data/raw/` -- needs rewrite for `.md`
- **Existing retriever:** Monolithic `app/rag.py` with hybrid search but no reranker, no filtering, no HyDE
- **No tests directory** exists yet

## Markdown Heading Patterns Per Document

| Document | Chuong | Dieu |
|----------|--------|------|
| Luat Dat Dai 2024 | `# Chuong I` | `### Dieu N.` |
| ND 102 | `# Chuong I` | `# Dieu N.` |
| ND 103 | `# Chuong II` / `## Chuong I` | `## Dieu N.` / `#### Dieu N.` |
| ND 151 | (mixed) | (mixed headings) |
| ND 226 | (no chapters, flat) | `**Dieu N.**` |
| NQ 254 | `## Chuong I` | `#### Dieu N.` / `**Dieu N.**` |
| ND 49 | `# Chuong I` | `## Dieu N.` / `### Dieu N.` / `**Dieu N.**` |
| ND 71 | `### Chuong I` | (mixed) |
| ND 88 | `### Chuong I` / `# Chuong II` | (mixed) |

**Key insight:** Cannot rely on heading level. Must match by text content pattern regardless of `#` depth.

---

## Task 1: Project scaffolding

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_chunker.py` (empty placeholder)
- Create: `app/rag/__init__.py`

**Step 1: Create test directory and rag package**

```bash
mkdir -p tests
touch tests/__init__.py
mkdir -p app/rag
touch app/rag/__init__.py
```

**Step 2: Update requirements.txt**

Add `fastembed>=0.6.1` for the cross-encoder reranker. The existing `qdrant-client` and `google-genai` stay.

Edit `requirements.txt` to append:

```
fastembed>=0.6.1
```

**Step 3: Commit**

```bash
git add tests/ app/rag/ requirements.txt
git commit -m "scaffold: add tests dir, app/rag package, fastembed dep"
```

---

## Task 2: Markdown chunker -- core parsing

**Files:**
- Create: `scripts/chunk_md.py`
- Create: `tests/test_chunker.py`

This replaces `scripts/chunk_documents.py`. Reads `.md` files from `data/`, not `data/raw/`.

**Step 1: Write failing tests for Dieu detection**

```python
# tests/test_chunker.py
"""Tests for the markdown legal document chunker."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.chunk_md import parse_dieu_heading, find_all_dieu, find_chapters


class TestDieuHeadingParsing:
    """Dieu headings appear in multiple markdown formats across documents."""

    def test_heading_hash_three(self):
        line = "### Điều 1. Phạm vi điều chỉnh"
        result = parse_dieu_heading(line)
        assert result == ("Điều 1", "Phạm vi điều chỉnh")

    def test_heading_hash_two(self):
        line = "## Điều 4. Diện tích đất tính tiền sử dụng đất"
        result = parse_dieu_heading(line)
        assert result == ("Điều 4", "Diện tích đất tính tiền sử dụng đất")

    def test_heading_hash_one(self):
        line = "# Điều 15. Quy định trình tự, thủ tục hành chính về đất đai"
        result = parse_dieu_heading(line)
        assert result == ("Điều 15", "Quy định trình tự, thủ tục hành chính về đất đai")

    def test_heading_hash_four(self):
        line = "#### Điều 1. Phạm vi điều chỉnh"
        result = parse_dieu_heading(line)
        assert result == ("Điều 1", "Phạm vi điều chỉnh")

    def test_bold_dieu(self):
        line = "**Điều 1. Sửa đổi, bổ sung một số điều**"
        result = parse_dieu_heading(line)
        assert result == ("Điều 1", "Sửa đổi, bổ sung một số điều")

    def test_non_dieu_line(self):
        line = "Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam"
        result = parse_dieu_heading(line)
        assert result is None

    def test_dieu_in_body_text_ignored(self):
        line = "theo quy định tại Điều 78 của Luật Đất đai"
        result = parse_dieu_heading(line)
        assert result is None


class TestFindChapters:

    def test_finds_hash_chapter(self):
        text = "# Chương I\n## QUY ĐỊNH CHUNG\n\nSome content"
        chapters = find_chapters(text)
        assert len(chapters) == 1
        assert chapters[0]["number"] == "I"

    def test_finds_multiple_hash_levels(self):
        text = "### Chương I\n### QUY ĐỊNH CHUNG\n\n# Chương II\nContent"
        chapters = find_chapters(text)
        assert len(chapters) == 2


class TestFindAllDieu:

    def test_basic_document(self):
        text = """# Chương I
## QUY ĐỊNH CHUNG

### Điều 1. Phạm vi điều chỉnh

Luật này quy định về chế độ sở hữu đất đai.

### Điều 2. Đối tượng áp dụng

1. Cơ quan nhà nước.
2. Người sử dụng đất.
"""
        results = find_all_dieu(text, "Test Doc")
        assert len(results) == 2
        assert results[0]["dieu"] == "Điều 1"
        assert results[0]["dieu_title"] == "Phạm vi điều chỉnh"
        assert "Luật này quy định" in results[0]["content"]
        assert results[1]["dieu"] == "Điều 2"

    def test_bold_dieu_format(self):
        text = """**Điều 1. Sửa đổi bổ sung**

1. Nội dung sửa đổi.

**Điều 2. Hiệu lực thi hành**

Nghị định này có hiệu lực.
"""
        results = find_all_dieu(text, "Test Doc")
        assert len(results) == 2

    def test_mixed_heading_levels(self):
        text = """## Điều 1. Phạm vi

Nội dung 1.

### Điều 3. Bồi thường

Nội dung 3.

**Điều 7. Bảng giá đất**

Nội dung 7.
"""
        results = find_all_dieu(text, "Test Doc")
        assert len(results) == 3
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_chunker.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'scripts.chunk_md'`

**Step 3: Implement core parsing functions**

```python
# scripts/chunk_md.py
"""
Chunk markdown legal documents by Dieu (Article) and Khoan (Clause).
Reads .md files from data/, outputs chunks as JSON to data/chunks/.

Handles inconsistent markdown heading levels across documents:
- ### Điều N. Title
- ## Điều N. Title
- # Điều N. Title
- #### Điều N. Title
- **Điều N. Title**
"""
import json
import os
import re
import sys
import unicodedata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "data", "chunks")

# Matches Dieu in any markdown heading format or bold
# Group 1: "Điều N", Group 2: title after the dot
DIEU_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+|\*\*)(Điều\s+\d+)\.\s*(.+?)(?:\*\*)?$"
)

# Matches Chuong in any heading format
CHUONG_RE = re.compile(
    r"^#{1,6}\s+Chương\s+([IVXLCDM]+|\d+)\s*$", re.IGNORECASE
)

# Khoan: "1. ", "2. " at line start
KHOAN_RE = re.compile(r"^(\d+)\.\s+", re.MULTILINE)

# Document name mapping
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
}


def get_doc_name(filename: str) -> str:
    """Map filename to canonical document name."""
    base = os.path.splitext(filename)[0]
    base_nfc = unicodedata.normalize("NFC", base)
    return DOC_NAME_MAP.get(base_nfc, DOC_NAME_MAP.get(base, base_nfc))


def parse_dieu_heading(line: str) -> tuple[str, str] | None:
    """Parse a single line as a Dieu heading. Returns (dieu_label, title) or None."""
    m = DIEU_HEADING_RE.match(line.strip())
    if m:
        return (m.group(1), m.group(2).strip().rstrip("*"))
    return None


def find_chapters(text: str) -> list[dict]:
    """Find all Chuong positions and their titles."""
    chapters = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = CHUONG_RE.match(line.strip())
        if m:
            # Title is typically the next non-empty line
            title = ""
            for j in range(i + 1, min(i + 3, len(lines))):
                candidate = lines[j].strip().lstrip("#").strip()
                if candidate and len(candidate) > 3:
                    title = candidate
                    break
            chapters.append({
                "number": m.group(1),
                "title": title,
                "line_idx": i,
            })
    return chapters


def _get_chapter_at_line(chapters: list[dict], line_idx: int) -> str:
    """Get the chapter label for a given line index."""
    current = ""
    for ch in chapters:
        if ch["line_idx"] <= line_idx:
            current = f"Chuong {ch['number']}: {ch['title']}"
        else:
            break
    return current


def find_all_dieu(text: str, doc_name: str) -> list[dict]:
    """Find all Dieu in text and extract their content.
    Returns list of chunk dicts with doc_name, chapter, dieu, dieu_title, content, metadata_str.
    """
    lines = text.split("\n")
    chapters = find_chapters(text)

    # Find all Dieu line positions
    dieu_positions = []
    for i, line in enumerate(lines):
        parsed = parse_dieu_heading(line)
        if parsed:
            dieu_positions.append((i, parsed[0], parsed[1]))

    if not dieu_positions:
        return [{
            "doc_name": doc_name,
            "chapter": "",
            "dieu": "",
            "dieu_title": "",
            "content": text[:3000],
            "metadata_str": doc_name,
        }]

    chunks = []
    for idx, (line_idx, dieu_label, dieu_title) in enumerate(dieu_positions):
        # Content runs from this Dieu heading to the next Dieu heading
        start = line_idx
        if idx + 1 < len(dieu_positions):
            end = dieu_positions[idx + 1][0]
        else:
            end = len(lines)

        content = "\n".join(lines[start:end]).strip()
        chapter = _get_chapter_at_line(chapters, line_idx)

        chunks.append({
            "doc_name": doc_name,
            "chapter": chapter,
            "dieu": dieu_label,
            "dieu_title": dieu_title,
            "content": content,
            "metadata_str": f"{doc_name} | {chapter} | {dieu_label}. {dieu_title}".strip(" |"),
        })

    return chunks
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_chunker.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add scripts/chunk_md.py tests/test_chunker.py
git commit -m "feat: markdown chunker with Dieu parsing across heading formats"
```

---

## Task 3: Markdown chunker -- Khoan splitting and size limits

**Files:**
- Modify: `scripts/chunk_md.py`
- Modify: `tests/test_chunker.py`

**Step 1: Add failing tests for Khoan splitting**

Append to `tests/test_chunker.py`:

```python
from scripts.chunk_md import split_long_dieu


class TestSplitLongDieu:

    def test_short_dieu_unchanged(self):
        chunk = {
            "doc_name": "Test",
            "chapter": "Chuong I",
            "dieu": "Điều 1",
            "dieu_title": "Title",
            "content": "Short content here.",
            "metadata_str": "Test | Chuong I | Điều 1. Title",
        }
        result = split_long_dieu(chunk, max_size=2000)
        assert len(result) == 1
        assert result[0]["content"] == "Short content here."

    def test_splits_by_khoan(self):
        header = "### Điều 5. Dài\n\n"
        body = ""
        for i in range(1, 6):
            body += f"{i}. Nội dung khoản {i} " + ("x " * 200) + "\n\n"
        chunk = {
            "doc_name": "Test",
            "chapter": "",
            "dieu": "Điều 5",
            "dieu_title": "Dài",
            "content": header + body,
            "metadata_str": "Test | Điều 5. Dài",
        }
        result = split_long_dieu(chunk, max_size=500)
        assert len(result) > 1
        for r in result:
            assert r["dieu"] == "Điều 5"
            assert "khoan" in r or "content" in r

    def test_fallback_size_split(self):
        # One giant Khoan with no sub-clauses
        content = "### Điều 99. Giant\n\n1. " + ("word " * 2000)
        chunk = {
            "doc_name": "Test",
            "chapter": "",
            "dieu": "Điều 99",
            "dieu_title": "Giant",
            "content": content,
            "metadata_str": "Test | Điều 99. Giant",
        }
        result = split_long_dieu(chunk, max_size=1500)
        assert len(result) > 1
        # All parts should reference the same Dieu
        for r in result:
            assert r["dieu"] == "Điều 99"
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_chunker.py::TestSplitLongDieu -v
```

Expected: FAIL -- `ImportError: cannot import name 'split_long_dieu'`

**Step 3: Implement split_long_dieu**

Add to `scripts/chunk_md.py`:

```python
def split_long_dieu(chunk: dict, max_size: int = 2000) -> list[dict]:
    """Split a long Dieu chunk by Khoan, falling back to size-based splits."""
    content = chunk["content"]
    if len(content) <= max_size:
        return [chunk]

    # Try splitting by Khoan
    khoan_matches = list(KHOAN_RE.finditer(content))
    if len(khoan_matches) >= 2:
        return _split_by_khoan(chunk, khoan_matches, max_size)

    # Fallback: fixed-size split
    return _split_by_size(chunk, max_size)


def _split_by_khoan(chunk: dict, khoan_matches: list, max_size: int) -> list[dict]:
    """Split content at Khoan boundaries."""
    content = chunk["content"]
    # Header = everything before first Khoan
    header = content[:khoan_matches[0].start()].strip()
    results = []

    for i, match in enumerate(khoan_matches):
        khoan_num = match.group(1)
        start = match.start()
        end = khoan_matches[i + 1].start() if i + 1 < len(khoan_matches) else len(content)
        khoan_text = content[start:end].strip()

        full_text = f"{header}\n\n{khoan_text}" if header else khoan_text

        # If still too long, split by size
        if len(full_text) > max_size + 500:
            sub_chunks = _split_by_size(
                {**chunk, "content": full_text, "khoan": khoan_num},
                max_size,
            )
            results.extend(sub_chunks)
        else:
            results.append({
                **chunk,
                "content": full_text,
                "khoan": khoan_num,
                "metadata_str": f"{chunk['metadata_str']} | Khoan {khoan_num}",
            })

    return results


def _split_by_size(chunk: dict, max_size: int = 1500, overlap: int = 200) -> list[dict]:
    """Fixed-size split with overlap at sentence boundaries."""
    content = chunk["content"]
    parts = []
    start = 0
    part_num = 1

    while start < len(content):
        end = start + max_size
        if end < len(content):
            for sep in ["\n\n", "\n", ". ", "; "]:
                last_sep = content.rfind(sep, start + max_size // 2, end)
                if last_sep > 0:
                    end = last_sep + len(sep)
                    break

        text = content[start:end].strip()
        if text:
            parts.append({
                **chunk,
                "content": text,
                "metadata_str": f"{chunk['metadata_str']} (phan {part_num})",
            })
            part_num += 1

        start = end - overlap if end < len(content) else end

    return parts
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_chunker.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add scripts/chunk_md.py tests/test_chunker.py
git commit -m "feat: chunker Khoan splitting and size-based fallback"
```

---

## Task 4: Markdown chunker -- clean_text and main() entrypoint

**Files:**
- Modify: `scripts/chunk_md.py`

**Step 1: Add clean_text and main**

Add to `scripts/chunk_md.py`:

```python
def clean_md_text(text: str) -> str:
    """Clean markdown text: remove signature blocks, page numbers, artifacts."""
    # Remove digital signature metadata blocks at the top
    text = re.sub(
        r"^(?:VGP\s.*\n)?Người ký:.*?\nEmail:.*?\nCơ quan:.*?\nThời gian ký:.*?\n",
        "", text, flags=re.MULTILINE
    )
    # Remove markdown table rows that are just headers/separators
    text = re.sub(r"^\|[-\s|:]+\|\s*$", "", text, flags=re.MULTILINE)
    # Remove page number lines (standalone small numbers)
    text = re.sub(r"^\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_file(filepath: str) -> list[dict]:
    """Process a single .md file into chunks."""
    filename = os.path.basename(filepath)
    doc_name = get_doc_name(filename)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_md_text(text)
    raw_chunks = find_all_dieu(text, doc_name)

    # Split long chunks
    final_chunks = []
    for chunk in raw_chunks:
        final_chunks.extend(split_long_dieu(chunk))

    return final_chunks


def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    md_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".md")
    )
    if not md_files:
        print(f"Khong tim thay file .md nao trong {DATA_DIR}")
        sys.exit(1)

    all_chunks = []
    print(f"Xu ly {len(md_files)} file .md:")

    for md_file in md_files:
        filepath = os.path.join(DATA_DIR, md_file)
        chunks = process_file(filepath)
        all_chunks.extend(chunks)
        print(f"  - {md_file}: {len(chunks)} chunks")

    # Assign stable IDs
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    # Save combined JSON
    output_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTong cong: {len(all_chunks)} chunks")
    print(f"Da luu vao: {output_path}")

    # Stats
    by_doc = {}
    for chunk in all_chunks:
        by_doc.setdefault(chunk["doc_name"], []).append(chunk)
    print("\nThong ke:")
    for doc_name, doc_chunks in sorted(by_doc.items()):
        sizes = [len(c["content"]) for c in doc_chunks]
        print(f"  {doc_name}: {len(doc_chunks)} chunks, avg {sum(sizes)//len(sizes)} chars")


if __name__ == "__main__":
    main()
```

**Step 2: Run the chunker on real data**

```bash
python scripts/chunk_md.py
```

Expected: Processes 9 .md files, prints chunk counts per document, saves `data/chunks/all_chunks.json`.

Verify the output makes sense: total chunks should be in the range of 500-800 given ~543 Dieu plus splits.

**Step 3: Commit**

```bash
git add scripts/chunk_md.py data/chunks/all_chunks.json
git commit -m "feat: complete markdown chunker with clean/split/main"
```

---

## Task 5: Qdrant index builder -- rebuild for new chunks

**Files:**
- Modify: `scripts/build_index.py`

The existing `build_index.py` already works with Qdrant + Gemini embeddings + sparse vectors. Adjustments needed:

1. Read chunks from the new `all_chunks.json` produced by `chunk_md.py`
2. Add `chunk_id` to payload for parent-child lookup
3. Add `doc_id` payload field (normalized short key like `ldd2024`, `nd102`, etc.) for filtering
4. Index `doc_id` as keyword field

**Step 1: Read current build_index.py** (already read above)

**Step 2: Modify to add doc_id and chunk_id to payload**

In the payload section of `build_index.py` (around line 196-210), add:

```python
# In the points building loop, update payload to include:
payload={
    "text": chunk.get("content", ""),
    "doc_name": chunk.get("doc_name", ""),
    "doc_id": _to_doc_id(chunk.get("doc_name", "")),
    "chapter": chunk.get("chapter", ""),
    "dieu": chunk.get("dieu", ""),
    "dieu_title": chunk.get("dieu_title", ""),
    "khoan": chunk.get("khoan", ""),
    "metadata_str": chunk.get("metadata_str", ""),
    "chunk_id": chunk.get("chunk_id", i),
},
```

Add the `_to_doc_id` helper:

```python
DOC_ID_MAP = {
    "luat dat dai": "ldd2024",
    "102": "nd102",
    "103": "nd103",
    "151": "nd151",
    "226": "nd226",
    "254": "nq254",
    "49": "nd49",
    "71": "nd71",
    "88": "nd88",
}

def _to_doc_id(doc_name: str) -> str:
    """Convert doc_name to short filterable ID."""
    name_lower = doc_name.lower()
    for key, doc_id in DOC_ID_MAP.items():
        if key in name_lower:
            return doc_id
    return name_lower.replace(" ", "_")[:20]
```

Add payload indexes for `doc_id`:

```python
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="doc_id",
    field_schema=PayloadSchemaType.KEYWORD,
)
```

**Step 3: Run the index builder with --fresh flag**

```bash
python scripts/build_index.py --fresh
```

Expected: Embeds all chunks, creates Qdrant collection with dense + sparse vectors.

**Step 4: Commit**

```bash
git add scripts/build_index.py
git commit -m "feat: index builder adds doc_id, chunk_id to payload"
```

---

## Task 6: RAG package -- knowledge_store.py (Qdrant client wrapper)

**Files:**
- Create: `app/rag/knowledge_store.py`

This is a thin wrapper around QdrantClient used by the retriever. It does not load/embed documents (that is done by `scripts/build_index.py`). It only provides search methods at runtime.

**Step 1: Write the module**

```python
# app/rag/knowledge_store.py
"""Qdrant client wrapper for search operations at runtime."""
import json
import logging
import os
import re
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, Fusion, FusionQuery,
    MatchAny, MatchValue, Prefetch, SparseVector,
)

from app.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, PROJECT_ROOT

logger = logging.getLogger(__name__)

VOCAB_FILE = os.path.join(PROJECT_ROOT, "data", "vocab.json")


class KnowledgeStore:
    """Runtime search interface to the Qdrant vector store."""

    def __init__(self):
        self._client: QdrantClient | None = None
        self._vocab: dict[str, int] = {}
        self._load_vocab()

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            if QDRANT_URL and QDRANT_URL != ":memory:":
                self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
            else:
                self._client = QdrantClient(location=":memory:")
        return self._client

    def _load_vocab(self):
        if os.path.exists(VOCAB_FILE):
            with open(VOCAB_FILE, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)

    @property
    def has_sparse(self) -> bool:
        return len(self._vocab) > 0

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(
            r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
            " ", text,
        )
        return [w for w in text.split() if len(w) > 1]

    def _text_to_sparse(self, text: str) -> SparseVector:
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        indices, values = [], []
        for token, count in sorted(counts.items()):
            if token in self._vocab:
                indices.append(self._vocab[token])
                values.append(float(count))
        return SparseVector(indices=indices, values=values)

    def search_hybrid(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 20,
        doc_ids: list[str] | None = None,
        dieu: str | None = None,
    ) -> list[dict]:
        """Hybrid dense+sparse search with RRF fusion. Returns list of payload dicts with score."""
        client = self._get_client()

        try:
            info = client.get_collection(QDRANT_COLLECTION)
            if info.points_count == 0:
                return []
        except Exception:
            return []

        # Build optional filter
        qdrant_filter = self._build_filter(doc_ids=doc_ids, dieu=dieu)

        sparse_vec = self._text_to_sparse(query_text)
        prefetch_limit = min(top_k * 3, info.points_count)

        prefetches = [
            Prefetch(query=query_vector, using="dense", limit=prefetch_limit),
        ]
        if sparse_vec.indices:
            prefetches.append(
                Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
            )

        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=prefetches,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {**point.payload, "score": point.score}
            for point in results
        ]

    def _build_filter(
        self,
        doc_ids: list[str] | None = None,
        dieu: str | None = None,
    ) -> Filter | None:
        conditions = []
        if doc_ids:
            conditions.append(
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
            )
        if dieu:
            conditions.append(
                FieldCondition(key="dieu", match=MatchValue(value=dieu))
            )
        if not conditions:
            return None
        return Filter(must=conditions)
```

**Step 2: Commit**

```bash
git add app/rag/knowledge_store.py
git commit -m "feat: KnowledgeStore runtime Qdrant wrapper with hybrid search and filtering"
```

---

## Task 7: RAG package -- query_rewriter.py

**Files:**
- Create: `app/rag/query_rewriter.py`

Extracts the existing query rewriting logic from `app/rag.py` into a dedicated module with improvements:
- Extracts doc/dieu references for metadata filtering
- Returns both rewritten query and extracted filter hints

**Step 1: Write the module**

```python
# app/rag/query_rewriter.py
"""Query rewriting and filter extraction for legal RAG."""
import logging
import re

from app import llm

logger = logging.getLogger(__name__)

# Maps user query patterns to doc_id filter values
DOC_PATTERNS = {
    r"(?:luật đất đai|luat dat dai|luật\s*số\s*31)": ["ldd2024"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*71": ["nd71"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*88": ["nd88"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*102": ["nd102"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*103": ["nd103"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*151": ["nd151"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*226": ["nd226"],
    r"(?:nghị quyết|nghi quyet|nq)\s*254": ["nq254"],
    r"(?:nghị định|nghi dinh|nđ|nd)\s*49": ["nd49"],
}

DIEU_PATTERN = re.compile(r"(?:điều|dieu|đ\.?)\s*(\d+)", re.IGNORECASE)


def extract_filters(query: str) -> dict:
    """Extract metadata filter hints from query text.
    Returns {"doc_ids": [...], "dieu": "Điều N" or None}.
    """
    query_lower = query.lower()
    doc_ids = []
    for pattern, ids in DOC_PATTERNS.items():
        if re.search(pattern, query_lower):
            doc_ids.extend(ids)

    dieu = None
    dieu_match = DIEU_PATTERN.search(query_lower)
    if dieu_match:
        dieu = f"Điều {dieu_match.group(1)}"

    return {
        "doc_ids": list(set(doc_ids)) if doc_ids else None,
        "dieu": dieu,
    }


async def rewrite_query(query: str, history: list[dict] | None = None) -> str:
    """Rewrite query using conversation context. Returns original if no history or on error."""
    if not history or len(history) < 2:
        return query

    recent = history[-6:]
    context = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in recent)

    prompt = f"""Viet lai cau hoi sau de ro rang hon, giai quyet dai tu va tham chieu ngam.
Giu nguyen y dinh goc. Tra ve CHI cau hoi da viet lai, khong giai thich.

Lich su hoi thoai gan day:
{context}

Cau hoi hien tai: {query}

Cau hoi da viet lai:"""

    try:
        rewritten = await llm.generate(prompt, model="flash", temperature=0.0, max_tokens=200)
        rewritten = rewritten.strip().strip('"').strip("'")
        if 10 < len(rewritten) < 500:
            return rewritten
    except Exception:
        pass
    return query
```

**Step 2: Commit**

```bash
git add app/rag/query_rewriter.py
git commit -m "feat: query rewriter with filter extraction from legal references"
```

---

## Task 8: RAG package -- reranker.py (cross-encoder)

**Files:**
- Create: `app/rag/reranker.py`
- Create: `tests/test_reranker.py`

Uses fastembed's TextCrossEncoder with `BAAI/bge-reranker-v2-m3` (multilingual, supports Vietnamese).

**Step 1: Write failing test**

```python
# tests/test_reranker.py
"""Tests for the cross-encoder reranker."""
import asyncio
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFastEmbedReranker:

    @pytest.fixture(scope="class")
    def reranker(self):
        from app.rag.reranker import CrossEncoderReranker
        return CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")

    def test_rerank_returns_top_k(self, reranker):
        candidates = [
            {"content": "Điều 1 quy định về phạm vi điều chỉnh của luật đất đai", "score": 0.5, "metadata": {}},
            {"content": "Thời tiết hôm nay trời nắng đẹp", "score": 0.8, "metadata": {}},
            {"content": "Luật đất đai 2024 quy định quyền sử dụng đất", "score": 0.3, "metadata": {}},
        ]
        result = asyncio.get_event_loop().run_until_complete(
            reranker.rerank("quy định về quyền sử dụng đất", candidates, top_k=2)
        )
        assert len(result) == 2
        # The legal content should score higher than weather
        assert "thời tiết" not in result[0]["content"].lower()

    def test_rerank_empty_list(self, reranker):
        result = asyncio.get_event_loop().run_until_complete(
            reranker.rerank("test query", [], top_k=5)
        )
        assert result == []
```

**Step 2: Implement the reranker**

```python
# app/rag/reranker.py
"""Cross-encoder reranker using FastEmbed ONNX inference."""
import asyncio
import logging
import math

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieval candidates using a cross-encoder model.
    Runs locally via ONNX -- no API calls.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self._model = TextCrossEncoder(model_name=model_name)
        logger.info("CrossEncoderReranker loaded (model=%s)", model_name)

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Score each (query, doc) pair and return top_k sorted by cross-encoder score."""
        if not candidates:
            return []

        documents = [c["content"] for c in candidates]

        # Run ONNX inference off the event loop
        raw_scores = await asyncio.to_thread(
            lambda: list(self._model.rerank(query, documents))
        )

        # Sigmoid normalization for bge-reranker (returns logits)
        scores = [1 / (1 + math.exp(-s)) for s in raw_scores]

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked = []
        for ce_score, candidate in scored[:top_k]:
            reranked.append({**candidate, "score": ce_score})

        logger.info(
            "Reranked %d -> %d (top scores: %s)",
            len(candidates), len(reranked),
            [f"{s:.3f}" for s, _ in scored[:top_k]],
        )
        return reranked
```

**Step 3: Run tests**

```bash
pip install fastembed>=0.6.1
python -m pytest tests/test_reranker.py -v
```

Expected: PASS (first run downloads the model, ~100MB)

**Step 4: Commit**

```bash
git add app/rag/reranker.py tests/test_reranker.py
git commit -m "feat: cross-encoder reranker with bge-reranker-v2-m3"
```

---

## Task 9: RAG package -- hyde.py (Hypothetical Document Embeddings)

**Files:**
- Create: `app/rag/hyde.py`

HyDE generates a hypothetical legal answer, embeds it, and uses that embedding for retrieval. This bridges the gap between colloquial user queries and formal legal language.

**Step 1: Write the module**

```python
# app/rag/hyde.py
"""HyDE: Hypothetical Document Embeddings for legal RAG.
Generates a hypothetical legal answer, embeds it, and uses that
embedding for dense retrieval instead of the raw query embedding.
"""
import logging

from app import llm

logger = logging.getLogger(__name__)

HYDE_PROMPT = """Ban la chuyen gia luat dat dai Viet Nam. Hay viet mot doan ngan (100-150 tu)
tra loi cau hoi sau nhu the ban dang trich dan tu van ban luat. Su dung ngon ngu phap ly
chinh thuc, trich dan so dieu, khoan neu co the. Khong can chinh xac, chi can giong van ban luat.

Cau hoi: {query}

Tra loi (van ban phap ly):"""


async def generate_hyde_passage(query: str) -> str | None:
    """Generate a hypothetical legal passage for the query."""
    try:
        passage = await llm.generate(
            HYDE_PROMPT.format(query=query),
            model="flash",
            temperature=0.3,
            max_tokens=300,
        )
        passage = passage.strip()
        if len(passage) > 30:
            logger.info("HyDE passage generated (%d chars)", len(passage))
            return passage
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)
    return None


async def get_hyde_embedding(query: str) -> list[float] | None:
    """Generate hypothetical passage and return its embedding."""
    passage = await generate_hyde_passage(query)
    if not passage:
        return None
    try:
        embeddings = llm.embed([passage])
        return embeddings[0]
    except Exception as e:
        logger.warning("HyDE embedding failed: %s", e)
        return None
```

**Step 2: Commit**

```bash
git add app/rag/hyde.py
git commit -m "feat: HyDE hypothetical document embeddings for legal queries"
```

---

## Task 10: RAG package -- retriever.py (main orchestrator)

**Files:**
- Create: `app/rag/retriever.py`

This is the main retrieval pipeline that ties everything together:
1. Extract filters from query
2. Rewrite query (if conversation history)
3. Generate HyDE embedding (optional)
4. Hybrid search via KnowledgeStore
5. Cross-encoder rerank
6. Relevance threshold filter
7. Return results

**Step 1: Write the module**

```python
# app/rag/retriever.py
"""Main retrieval pipeline orchestrator."""
import logging
import re

from app import llm
from app.config import RAG_TOP_K, RAG_RERANK_CANDIDATES
from app.rag.knowledge_store import KnowledgeStore
from app.rag.query_rewriter import extract_filters, rewrite_query
from app.rag.reranker import CrossEncoderReranker
from app.rag import hyde

logger = logging.getLogger(__name__)

# Config
RELEVANCE_THRESHOLD = 0.15  # Cross-encoder score floor (sigmoid: 0.5 = decision boundary)
USE_HYDE = True
USE_RERANKER = True


class Retriever:
    """Full retrieval pipeline: rewrite -> HyDE -> hybrid search -> rerank -> threshold."""

    def __init__(self):
        self._store = KnowledgeStore()
        self._reranker: CrossEncoderReranker | None = None
        if USE_RERANKER:
            try:
                self._reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning("Reranker init failed, running without: %s", e)

    async def retrieve(
        self,
        query: str,
        history: list[dict] | None = None,
        top_k: int = RAG_TOP_K,
    ) -> dict:
        """Run the full retrieval pipeline.
        Returns {
            "chunks": list[dict],   # scored and sorted results
            "sources": list[dict],  # unique source references
            "rewritten_query": str | None,
            "filters_used": dict,
        }
        """
        # Step 1: Extract metadata filters from query
        filters = extract_filters(query)

        # Step 2: Rewrite query with conversation context
        search_query = await rewrite_query(query, history)
        rewritten = search_query if search_query != query else None

        # Step 3: Get embeddings
        # Try HyDE first, fall back to direct query embedding
        query_embedding = None
        if USE_HYDE:
            query_embedding = await hyde.get_hyde_embedding(search_query)

        if query_embedding is None:
            query_embedding = llm.embed([search_query])[0]

        # Step 4: Hybrid search
        fetch_k = RAG_RERANK_CANDIDATES if self._reranker else top_k
        candidates = self._store.search_hybrid(
            query_vector=query_embedding,
            query_text=search_query,
            top_k=fetch_k,
            doc_ids=filters.get("doc_ids"),
            dieu=filters.get("dieu"),
        )

        if not candidates:
            # Keyword fallback: retry without filters
            if filters.get("doc_ids") or filters.get("dieu"):
                candidates = self._store.search_hybrid(
                    query_vector=query_embedding,
                    query_text=search_query,
                    top_k=fetch_k,
                )

        if not candidates:
            return {
                "chunks": [],
                "sources": [],
                "rewritten_query": rewritten,
                "filters_used": filters,
            }

        # Step 5: Cross-encoder rerank
        if self._reranker and len(candidates) > top_k:
            try:
                candidates = await self._reranker.rerank(search_query, candidates, top_k * 2)
            except Exception as e:
                logger.warning("Reranker failed, using RRF order: %s", e)

        # Step 6: Relevance threshold (only with reranker -- RRF scores are not calibrated)
        if self._reranker:
            candidates = [c for c in candidates if c["score"] >= RELEVANCE_THRESHOLD]

        # Step 7: Trim to top_k
        chunks = candidates[:top_k]

        # Build sources
        sources = self._extract_sources(chunks)

        return {
            "chunks": chunks,
            "sources": sources,
            "rewritten_query": rewritten,
            "filters_used": filters,
        }

    @staticmethod
    def _extract_sources(chunks: list[dict]) -> list[dict]:
        """Extract unique source references from chunks."""
        sources = []
        seen = set()
        for chunk in chunks:
            key = f"{chunk.get('doc_name', '')}|{chunk.get('dieu', '')}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "doc_name": chunk.get("doc_name", ""),
                    "chapter": chunk.get("chapter", ""),
                    "dieu": chunk.get("dieu", ""),
                    "dieu_title": chunk.get("dieu_title", ""),
                })
        return sources


def build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks for LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata_str", "")
        content = chunk.get("text", chunk.get("content", ""))
        parts.append(f"[Nguon {i}: {source}]\n{content}")
    return "\n\n---\n\n".join(parts)
```

**Step 2: Commit**

```bash
git add app/rag/retriever.py
git commit -m "feat: retriever pipeline with HyDE, reranking, filtering, threshold"
```

---

## Task 11: RAG package -- __init__.py (public API)

**Files:**
- Modify: `app/rag/__init__.py`

**Step 1: Write the public API**

```python
# app/rag/__init__.py
"""RAG package public API."""
from app.rag.retriever import Retriever, build_context

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


async def search(query: str, history: list[dict] | None = None, top_k: int = 8) -> dict:
    """Convenience wrapper for retriever.retrieve()."""
    return await get_retriever().retrieve(query, history=history, top_k=top_k)
```

**Step 2: Commit**

```bash
git add app/rag/__init__.py
git commit -m "feat: rag package public API"
```

---

## Task 12: Wire retriever into chat.py

**Files:**
- Modify: `app/chat.py`

Replace calls to old `app.rag` module with new `app.rag` package.

**Step 1: Update imports and handle_message**

Key changes in `app/chat.py`:

```python
# Replace:
#   from app import db, llm, rag
#   from app.rag import rewrite_query, search, build_context, extract_sources
# With:
from app import db, llm
from app.rag import search
from app.rag.retriever import build_context
```

Update `handle_message` to use the new retriever API:

```python
async def handle_message(session_id: str, user_message: str) -> dict:
    db.create_session(session_id)
    history = db.get_messages(session_id)
    turn = db.get_turn_count(session_id) + 1
    db.add_message(session_id, turn, "user", user_message)

    complexity = llm.classify_complexity(user_message)
    model = "pro" if complexity == "complex" else "flash"

    # New retriever handles rewriting, HyDE, filtering, reranking internally
    history_dicts = _to_history_dicts(history)
    retrieval = await search(user_message, history=history_dicts)

    chunks = retrieval["chunks"]
    context = build_context(chunks)
    sources = retrieval["sources"]

    llm_history = _build_llm_context(history, session_id)
    prompt = _build_prompt(user_message, context)

    response = await llm.generate(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        history=llm_history,
        model=model,
        temperature=0.3,
        max_tokens=4000,
    )

    db.add_message(session_id, turn, "assistant", response, sources)

    if turn % SUMMARY_INTERVAL_TURNS == 0:
        await _generate_summary(session_id, history, turn)
    if turn == 1:
        await _generate_title(session_id, user_message, response)

    return {
        "answer": response,
        "sources": sources,
        "session_id": session_id,
    }
```

**Step 2: Delete old app/rag.py**

```bash
rm app/rag.py
```

The old monolithic module is fully replaced by `app/rag/` package.

**Step 3: Verify the server starts**

```bash
python -c "from app.rag import search; print('import OK')"
```

**Step 4: Commit**

```bash
git add app/chat.py
git rm app/rag.py
git commit -m "feat: wire new retriever into chat, remove old monolithic rag.py"
```

---

## Task 13: Config updates

**Files:**
- Modify: `app/config.py`

**Step 1: Add new RAG config values**

Add to `app/config.py`:

```python
# RAG
RAG_TOP_K = 8
RAG_RERANK_CANDIDATES = 20
RAG_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RAG_RELEVANCE_THRESHOLD = 0.15
RAG_USE_HYDE = True
RAG_USE_RERANKER = True
RAG_USE_QUERY_REWRITE = True
```

Update `retriever.py` to read from config instead of module-level constants.

**Step 2: Commit**

```bash
git add app/config.py app/rag/retriever.py
git commit -m "feat: externalize RAG config flags"
```

---

## Task 14: Integration test

**Files:**
- Create: `tests/test_retriever_integration.py`

**Step 1: Write integration test**

```python
# tests/test_retriever_integration.py
"""Integration test: chunks -> index -> retrieve.
Requires GEMINI_API_KEY and QDRANT_URL in .env.
Skip if not available.
"""
import asyncio
import os
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY"))
HAS_QDRANT = bool(os.getenv("QDRANT_URL")) and os.getenv("QDRANT_URL") != ":memory:"


@pytest.mark.skipif(not HAS_API_KEY or not HAS_QDRANT, reason="Needs GEMINI_API_KEY and QDRANT_URL")
class TestRetrieverIntegration:

    def test_basic_retrieval(self):
        from app.rag import search
        result = asyncio.get_event_loop().run_until_complete(
            search("quyền sử dụng đất là gì")
        )
        assert len(result["chunks"]) > 0
        assert any("đất" in c.get("text", c.get("content", "")).lower() for c in result["chunks"])

    def test_filter_by_document(self):
        from app.rag import search
        result = asyncio.get_event_loop().run_until_complete(
            search("Điều 5 Nghị định 102 quy định gì")
        )
        assert result["filters_used"]["doc_ids"] == ["nd102"]
        assert result["filters_used"]["dieu"] == "Điều 5"

    def test_empty_query_returns_results(self):
        from app.rag import search
        result = asyncio.get_event_loop().run_until_complete(
            search("bồi thường khi thu hồi đất")
        )
        assert len(result["chunks"]) > 0
```

**Step 2: Run**

```bash
python -m pytest tests/test_retriever_integration.py -v
```

Expected: PASS (if API keys configured)

**Step 3: Commit**

```bash
git add tests/test_retriever_integration.py
git commit -m "test: integration test for retriever pipeline"
```

---

## Task 15: End-to-end smoke test

**Files:** None new

**Step 1: Start the server**

```bash
uvicorn app.main:app --reload --port 8000
```

**Step 2: Test via curl**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Điều 5 Luật Đất Đai 2024 quy định gì?"}'
```

Expected: JSON response with `answer`, `sources`, `session_id`. The answer should cite Dieu 5 specifically.

**Step 3: Test filter extraction**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Nghị định 88 quy định gì về bồi thường?"}'
```

Expected: Response specifically about ND 88 content (not random documents).

**Step 4: Commit final state**

```bash
git add -A
git commit -m "chore: end-to-end smoke test verified"
```

---

## Summary of What Gets Built

| Component | File | What It Does |
|-----------|------|--------------|
| Markdown chunker | `scripts/chunk_md.py` | Parses 9 .md files, splits by Dieu/Khoan, outputs JSON |
| Index builder | `scripts/build_index.py` (modified) | Adds doc_id, chunk_id to Qdrant payload |
| Knowledge store | `app/rag/knowledge_store.py` | Qdrant runtime wrapper with hybrid search + payload filtering |
| Query rewriter | `app/rag/query_rewriter.py` | LLM rewrite + metadata filter extraction |
| Cross-encoder | `app/rag/reranker.py` | BAAI/bge-reranker-v2-m3 via FastEmbed ONNX |
| HyDE | `app/rag/hyde.py` | Generates hypothetical legal passage for embedding |
| Retriever | `app/rag/retriever.py` | Pipeline: rewrite -> HyDE -> search -> rerank -> threshold |
| Package API | `app/rag/__init__.py` | Clean public interface |

## Improvements Over Current Pipeline

1. **Cross-encoder reranking** -- neural reranker replaces naive metadata boost
2. **HyDE** -- bridges colloquial queries to formal legal language
3. **Payload filtering** -- explicit doc/dieu filters at Qdrant level (not post-hoc)
4. **Relevance threshold** -- drops low-quality results instead of always returning top_k
5. **Filter extraction** -- automatically detects "ND 88", "Dieu 5" in queries
6. **Fallback without filters** -- retries unfiltered if filtered search returns nothing
7. **Markdown-native chunking** -- reads .md directly, handles all heading formats
8. **Modular architecture** -- each concern in its own module, testable independently
