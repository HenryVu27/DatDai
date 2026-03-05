"""Tests for scripts/chunk_md.py"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.chunk_md import (
    parse_dieu_heading,
    find_chapters,
    find_all_dieu,
    split_long_dieu,
    clean_md_text,
)


# ── parse_dieu_heading ──────────────────────────────────────────────


class TestParseDieuHeading:
    def test_h3_heading(self):
        result = parse_dieu_heading("### Điều 1. Phạm vi điều chỉnh")
        assert result == ("Điều 1", "Phạm vi điều chỉnh")

    def test_h2_heading(self):
        result = parse_dieu_heading("## Điều 12. Đối tượng áp dụng")
        assert result == ("Điều 12", "Đối tượng áp dụng")

    def test_h1_heading(self):
        result = parse_dieu_heading("# Điều 100. Trách nhiệm quản lý đất đai")
        assert result == ("Điều 100", "Trách nhiệm quản lý đất đai")

    def test_h4_heading(self):
        result = parse_dieu_heading("#### Điều 3. Quy định về thu hồi đất")
        assert result == ("Điều 3", "Quy định về thu hồi đất")

    def test_bold_heading(self):
        result = parse_dieu_heading("**Điều 4. Giao đất cho thuê đất**")
        assert result == ("Điều 4", "Giao đất cho thuê đất")

    def test_bold_heading_long_title(self):
        result = parse_dieu_heading(
            "**Điều 5. Căn cứ tính tiền sử dụng đất, tiền thuê đất**"
        )
        assert result == (
            "Điều 5",
            "Căn cứ tính tiền sử dụng đất, tiền thuê đất",
        )

    def test_non_dieu_line_returns_none(self):
        assert parse_dieu_heading("Đây là một dòng bình thường") is None

    def test_empty_line_returns_none(self):
        assert parse_dieu_heading("") is None

    def test_mid_sentence_dieu_returns_none(self):
        line = "theo quy định tại Điều 78 của Luật Đất đai"
        assert parse_dieu_heading(line) is None

    def test_mid_sentence_reference_returns_none(self):
        line = "khoản 6 Điều 3, Điều 9, khoản 2 Điều 10"
        assert parse_dieu_heading(line) is None

    def test_chapter_heading_returns_none(self):
        assert parse_dieu_heading("# Chương I") is None

    def test_h5_heading(self):
        result = parse_dieu_heading("##### Điều 7. Bảng giá đất")
        assert result == ("Điều 7", "Bảng giá đất")

    def test_h6_heading(self):
        result = parse_dieu_heading("###### Điều 260. Điều khoản thi hành")
        assert result == ("Điều 260", "Điều khoản thi hành")

    def test_bold_with_period_inside(self):
        """Format: **Điều N.** Title after bold"""
        result = parse_dieu_heading("**Điều 3.** Quyết định này có hiệu lực kể từ ngày ký.")
        assert result is not None
        assert result[0] == "Điều 3"
        assert "Quyết định" in result[1]


# ── find_chapters ───────────────────────────────────────────────────


class TestFindChapters:
    def test_h1_chapter(self):
        text = "# Chương I\n# QUY ĐỊNH CHUNG\n\nSome content"
        chapters = find_chapters(text)
        assert len(chapters) == 1
        assert chapters[0]["number"] == "I"
        assert chapters[0]["title"] == "QUY ĐỊNH CHUNG"
        assert chapters[0]["line_idx"] == 0

    def test_h2_chapter(self):
        text = "## Chương II\n\n### MỘT SỐ CƠ CHẾ\n\nContent"
        chapters = find_chapters(text)
        assert len(chapters) == 1
        assert chapters[0]["number"] == "II"
        assert chapters[0]["title"] == "MỘT SỐ CƠ CHẾ"

    def test_h3_chapter(self):
        text = "### Chương III\nĐIỀU KHOẢN THI HÀNH\n"
        chapters = find_chapters(text)
        assert len(chapters) == 1
        assert chapters[0]["number"] == "III"
        assert chapters[0]["title"] == "ĐIỀU KHOẢN THI HÀNH"

    def test_multiple_chapters(self):
        text = (
            "# Chương I\n# QUY ĐỊNH CHUNG\n\nContent\n\n"
            "## Chương II\n\n## CHI TIẾT\n\nMore content\n\n"
            "# Chương XVI\nĐIỀU KHOẢN\n"
        )
        chapters = find_chapters(text)
        assert len(chapters) == 3
        assert chapters[0]["number"] == "I"
        assert chapters[1]["number"] == "II"
        assert chapters[2]["number"] == "XVI"

    def test_no_chapters(self):
        text = "Some text\nMore text\n"
        assert find_chapters(text) == []

    def test_title_strips_markdown_heading(self):
        """Title line may itself have # markers that should be stripped."""
        text = "# Chương IV\n## SỬA ĐỔI BỔ SUNG\n"
        chapters = find_chapters(text)
        assert chapters[0]["title"] == "SỬA ĐỔI BỔ SUNG"


# ── find_all_dieu ───────────────────────────────────────────────────


class TestFindAllDieu:
    def test_mixed_heading_formats(self):
        text = (
            "# Chương I\n# QUY ĐỊNH CHUNG\n\n"
            "### Điều 1. Phạm vi điều chỉnh\n\nNội dung điều 1.\n\n"
            "**Điều 2. Đối tượng áp dụng**\n\nNội dung điều 2.\n\n"
            "# Điều 3. Giải thích từ ngữ\n\nNội dung điều 3.\n"
        )
        chunks = find_all_dieu(text, "Test Doc")
        assert len(chunks) == 3
        assert chunks[0]["dieu"] == "Điều 1"
        assert chunks[1]["dieu"] == "Điều 2"
        assert chunks[2]["dieu"] == "Điều 3"
        assert "Test Doc" in chunks[0]["metadata_str"]
        assert "Chương I" in chunks[0]["chapter"]

    def test_no_dieu_returns_single_chunk(self):
        text = "This document has no Dieu headings at all.\nJust some text."
        chunks = find_all_dieu(text, "No Dieu Doc")
        assert len(chunks) == 1
        assert chunks[0]["doc_name"] == "No Dieu Doc"
        assert "no Dieu headings" in chunks[0]["content"]

    def test_content_spans_to_next_dieu(self):
        text = (
            "### Điều 1. First\n\nLine A\nLine B\n\n"
            "### Điều 2. Second\n\nLine C\n"
        )
        chunks = find_all_dieu(text, "Doc")
        assert "Line A" in chunks[0]["content"]
        assert "Line B" in chunks[0]["content"]
        assert "Line C" not in chunks[0]["content"]
        assert "Line C" in chunks[1]["content"]

    def test_chapter_tracking(self):
        text = (
            "# Chương I\nMỘT\n\n"
            "### Điều 1. In chapter I\n\nContent\n\n"
            "# Chương II\nHAI\n\n"
            "### Điều 2. In chapter II\n\nContent\n"
        )
        chunks = find_all_dieu(text, "Doc")
        assert "Chương I" in chunks[0]["chapter"]
        assert "Chương II" in chunks[1]["chapter"]


# ── split_long_dieu ─────────────────────────────────────────────────


class TestSplitLongDieu:
    def _make_chunk(self, content):
        return {
            "doc_name": "Test",
            "chapter": "Chương I: Test",
            "dieu": "Điều 1",
            "dieu_title": "Test title",
            "content": content,
            "metadata_str": "[Test] [Điều 1. Test title]",
        }

    def test_short_chunk_unchanged(self):
        chunk = self._make_chunk("Short content")
        result = split_long_dieu(chunk, max_size=2000)
        assert len(result) == 1
        assert result[0]["content"] == "Short content"

    def test_split_by_khoan(self):
        lines = ["### Điều 1. Test title", ""]
        # Create content with numbered Khoan; total exceeds max_size
        # but each individual Khoan + header fits within max_size
        for i in range(1, 6):
            lines.append(f"{i}. " + "Noi dung khoan day du. " * 15)
            lines.append("")

        chunk = self._make_chunk("\n".join(lines))
        # Total content ~2000 chars, max_size=500 forces Khoan split
        # Each Khoan ~350 chars + header ~25 chars = ~375 < 500
        result = split_long_dieu(chunk, max_size=500)
        assert len(result) == 5
        # Each sub-chunk should contain the header
        for r in result:
            assert "Điều 1" in r["content"]
        assert "Khoản 1" in result[0]["metadata_str"]
        assert "Khoản 5" in result[4]["metadata_str"]

    def test_split_by_khoan_then_by_size(self):
        """When Khoan chunks are still too large, they get further split by size."""
        lines = ["### Điều 1. Test title", ""]
        for i in range(1, 4):
            lines.append(f"{i}. " + "Nội dung khoản dài. " * 50)
            lines.append("")

        chunk = self._make_chunk("\n".join(lines))
        result = split_long_dieu(chunk, max_size=500)
        # Should be more than 3 since each Khoan is further split
        assert len(result) > 3
        # All pieces should reference Điều 1
        for r in result:
            assert "Điều 1" in r["metadata_str"]

    def test_fallback_size_split(self):
        # Content with no Khoan boundaries, just long text
        content = "### Điều 1. Test\n\n" + "Đây là nội dung rất dài. " * 200
        chunk = self._make_chunk(content)
        result = split_long_dieu(chunk, max_size=500)
        assert len(result) > 1
        # Each piece should have part number in metadata
        assert "Phần 1" in result[0]["metadata_str"]

    def test_single_khoan_uses_fallback(self):
        # Only 1 Khoan -- not enough to split by Khoan
        content = "### Điều 1. Test\n\n1. " + "Nội dung. " * 300
        chunk = self._make_chunk(content)
        result = split_long_dieu(chunk, max_size=500)
        # Should fall back to size-based splitting
        assert len(result) > 1
        assert "Phần" in result[0]["metadata_str"]


# ── clean_md_text ───────────────────────────────────────────────────


class TestCleanMdText:
    def test_remove_signature_block(self):
        text = (
            "VGP CỔNG THÔNG TIN ĐIỆN TỬ CHÍNH PHỦ\n"
            "Người ký: CỔNG THÔNG TIN ĐIỆN TỬ CHÍNH PHỦ\n"
            "Email: thongtinchinhphu@chinhphu.vn\n"
            "Cơ quan: VĂN PHÒNG CHÍNH PHỦ\n"
            "Thời gian ký: 28.10.2025 11:28:17 +07:00\n"
            "\n"
            "# NGHỊ ĐỊNH\n"
        )
        result = clean_md_text(text)
        assert "VGP" not in result
        assert "Người ký" not in result
        assert "NGHỊ ĐỊNH" in result

    def test_remove_table_separators(self):
        text = "| Header | Header2 |\n| --- | --- |\n| Data | Data2 |\n"
        result = clean_md_text(text)
        assert "| --- |" not in result
        assert "Data" in result

    def test_remove_standalone_page_numbers(self):
        text = "Some content\n\n42\n\nMore content\n"
        result = clean_md_text(text)
        assert "\n42\n" not in result
        assert "Some content" in result

    def test_collapse_blank_lines(self):
        text = "Line 1\n\n\n\n\nLine 2\n"
        result = clean_md_text(text)
        # Should have at most 2 consecutive blank lines (3 newlines)
        assert "\n\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_preserves_normal_content(self):
        text = "### Điều 1. Test\n\n1. Content here.\n2. More content.\n"
        result = clean_md_text(text)
        assert result.strip() == text.strip()
