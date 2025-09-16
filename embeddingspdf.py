import os
import pickle
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =========================
# CONFIG
# =========================
WORD_DIR = r"C:\Users\mayan\Desktop\newbot\data"  # your Word docs
OUTPUT_DIR = r"C:\Users\mayan\Desktop\newbot\models"

INDEX_FILE = os.path.join(OUTPUT_DIR, "word_embeddings.faiss")
METADATA_FILE = os.path.join(OUTPUT_DIR, "word_metadata.pkl")

CHUNK_SIZE = 500       # characters
CHUNK_OVERLAP = 100    # characters

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# UTILITIES
# =========================
def load_word_texts(word_dir: str) -> List[Dict]:
    word_docs = []

    for file_path in Path(word_dir).glob("*.docx"):
        doc = Document(file_path)
        full_text = ""
        for para in doc.paragraphs:
            text = para.text
            if text and text.strip():
                full_text += text.strip() + "\n"

        if not full_text.strip():
            continue

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        chunks = splitter.split_text(full_text)

        for idx, chunk in enumerate(chunks):
            if chunk and chunk.strip():
                word_docs.append({
                    "id": f"{file_path.stem}_chunk_{idx}",
                    "model": f"WORD:{file_path.name}",
                    "text": chunk.strip(),
                    "source_type": "word"
                })
    return word_docs


def create_index_and_save(docs: List[Dict], index_file: str, metadata_file: str):
    if not docs:
        print("❌ No Word documents found to embed!")
        return

    texts = [doc["text"].strip() for doc in docs if doc.get("text") and doc["text"].strip()]
    if not texts:
        print("❌ No valid text chunks to embed.")
        return

    print(f"Creating embeddings for {len(texts)} Word chunks...")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(docs, f)

    print(f"✅ Saved {index_file} and {metadata_file}, total chunks: {len(docs)}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Loading Word documents...")
    word_docs = load_word_texts(WORD_DIR)
    print(f"Total Word chunks: {len(word_docs)}")
    create_index_and_save(word_docs, INDEX_FILE, METADATA_FILE)
