"""
Extract text from all PDF files in the project root.
Handles both text-based and image-based (scanned) PDFs.
For image PDFs, uses Tesseract OCR with Vietnamese language pack.
Saves each PDF's text as a .txt file in data/raw/
"""
import os
import subprocess
import sys
import tempfile

import fitz  # PyMuPDF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = PROJECT_ROOT
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def extract_text_pdf(pdf_path: str) -> str:
    """Extract text from a text-based PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text.strip())
    doc.close()
    return "\n\n".join(pages_text)


def is_image_pdf(pdf_path: str) -> bool:
    """Check if PDF is image-based (scanned) rather than text-based."""
    doc = fitz.open(pdf_path)
    total_text = 0
    for page in doc:
        text = page.get_text("text").strip()
        # Ignore digital signature text
        if "Người ký" in text or "Email:" in text:
            continue
        total_text += len(text)
    doc.close()
    return total_text < 500


def ocr_pdf_tesseract(pdf_path: str, filename: str) -> str:
    """OCR a scanned PDF using Tesseract with Vietnamese language.
    Renders each page as image, runs tesseract, collects text.
    """
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    all_text = []

    for page_num in range(num_pages):
        if (page_num + 1) % 10 == 0 or page_num == 0:
            print(f"    Trang {page_num + 1}/{num_pages}...")

        page = doc[page_num]

        # Skip signature page (usually page 1 in signed PDFs)
        quick_text = page.get_text("text").strip()
        if "Người ký" in quick_text and page_num == 0:
            continue

        # Render page to image at 300 DPI for OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)

        # Write to temp file for tesseract
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pix.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Run tesseract with Vietnamese language
            result = subprocess.run(
                ["tesseract", tmp_path, "stdout", "-l", "vie", "--psm", "6"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            text = result.stdout.strip()
            if text:
                all_text.append(text)
        except subprocess.TimeoutExpired:
            print(f"    [TIMEOUT] Trang {page_num + 1}")
        except Exception as e:
            print(f"    [LOI] Trang {page_num + 1}: {e}")
        finally:
            os.unlink(tmp_path)

    doc.close()
    return "\n\n".join(all_text)


def process_pdf(pdf_path: str, filename: str) -> tuple[str, int]:
    """Process a single PDF file. Returns (text, page_count)."""
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    if is_image_pdf(pdf_path):
        print(f"  [{filename}] Scanned PDF ({page_count} trang) - OCR voi Tesseract...")
        text = ocr_pdf_tesseract(pdf_path, filename)
    else:
        print(f"  [{filename}] Text PDF ({page_count} trang) - trich xuat truc tiep...")
        text = extract_text_pdf(pdf_path)

    return text, page_count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    if not pdf_files:
        print("Khong tim thay file PDF nao.")
        sys.exit(1)

    print(f"Tim thay {len(pdf_files)} file PDF\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
        output_path = os.path.join(OUTPUT_DIR, txt_filename)

        # Skip if already extracted with substantial content
        if os.path.exists(output_path):
            existing_size = os.path.getsize(output_path)
            if existing_size > 5000:
                print(f"  [{pdf_file}] Da co ({existing_size:,} bytes) - bo qua")
                continue

        text, pages = process_pdf(pdf_path, pdf_file)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        size = len(text)
        print(f"  -> {txt_filename} ({size:,} chars)\n")

    print(f"\nHoan thanh! Text luu tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
