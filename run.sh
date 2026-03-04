#!/bin/bash
set -e

echo "=== Chatbot Luat Dat Dai ==="
echo ""

# Check if .env exists and has API key
if [ ! -f .env ]; then
    echo "Loi: Chua co file .env"
    echo "Tao file .env voi noi dung: OPENAI_API_KEY=sk-your-key-here"
    exit 1
fi

# Install dependencies
echo "[1/4] Cai dat thu vien..."
pip install -r requirements.txt -q

# Extract PDF
echo "[2/4] Trich xuat text tu PDF..."
python scripts/extract_pdf.py

# Chunk documents
echo "[3/4] Phan tach van ban thanh chunks..."
python scripts/chunk_documents.py

# Build index
echo "[4/4] Tao embeddings va luu vao ChromaDB..."
python scripts/build_index.py

echo ""
echo "=== Hoan thanh! Khoi dong server... ==="
echo "Mo trinh duyet tai: http://localhost:8000"
echo ""
uvicorn app.main:app --host 0.0.0.0 --port 8000
