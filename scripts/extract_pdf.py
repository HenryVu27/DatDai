"""
Extract text from all PDF files in the project root.
Saves each PDF's text as a .txt file in data/raw/
"""
import os
import sys
import fitz  # PyMuPDF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = PROJECT_ROOT
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def extract_pdf(pdf_path: str, output_path: str) -> int:
    """Extract text from a PDF file and save to .txt file.
    Returns the number of pages processed.
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"--- Trang {page_num + 1} ---\n{text}")

    doc.close()

    full_text = "\n\n".join(pages_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return len(pages_text)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("Khong tim thay file PDF nao trong thu muc goc.")
        sys.exit(1)

    print(f"Tim thay {len(pdf_files)} file PDF:")
    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        # Create output filename: replace .pdf with .txt
        txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
        output_path = os.path.join(OUTPUT_DIR, txt_filename)

        pages = extract_pdf(pdf_path, output_path)
        print(f"  - {pdf_file} -> {txt_filename} ({pages} trang)")

    print(f"\nDa luu tat ca text vao: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
