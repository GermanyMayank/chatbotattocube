import os
import pickle
from glob import glob
from typing import List, Dict
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# ----------------------------
# Configuration
# ----------------------------
JSON_DIR = r"C:\Users\mayan\Desktop\newbot\data"  # folder with JSON files
PDF_DIR = r"C:\Users\mayan\Desktop\newbot\data"    # folder with PDF files
OUTPUT_DIR = r"C:\Users\mayan\Desktop\newbot\models"

INDEX_FILE = os.path.join(OUTPUT_DIR, "embeddings.faiss")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.pkl")

CHUNK_SIZE = 300   # words per chunk
CHUNK_OVERLAP = 50 # overlap words between chunks

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Utility functions
# ----------------------------
def chunk_text(text: str, chunk_size=300, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_json_texts(json_dir: str) -> List[Dict]:
    import json
    texts = []
    for file in glob(os.path.join(json_dir, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    text = item.get("text", "")
                    if text:
                        texts.append({
                            "id": item.get("id", ""),
                            "model": item.get("model", ""),
                            "text": text
                        })
            elif isinstance(data, dict):
                text = data.get("text", "")
                if text:
                    texts.append({
                        "id": data.get("id", ""),
                        "model": data.get("model", ""),
                        "text": text
                    })
    return texts

def load_pdf_texts(pdf_dir: str) -> List[Dict]:
    texts = []
    pdf_files = glob(os.path.join(pdf_dir, "*.pdf"))
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        # chunk PDF text
        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            texts.append({
                "id": f"{Path(pdf_file).stem}_chunk_{i}",
                "model": f"PDF:{Path(pdf_file).name}",
                "text": chunk
            })
    return texts

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("Loading JSON data...")
    json_texts = load_json_texts(JSON_DIR)

    print("Loading PDF data...")
    pdf_texts = load_pdf_texts(PDF_DIR)

    all_docs = json_texts + pdf_texts
    print(f"Total documents/chunks to embed: {len(all_docs)}")

    # Create embeddings
    print("Creating embeddings...")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    corpus_texts = [doc["text"] for doc in all_docs]
    embeddings = embedder.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)

    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save FAISS index and metadata
    print("Saving FAISS index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(all_docs, f)

    print("✅ Embeddings and FAISS index saved!")
