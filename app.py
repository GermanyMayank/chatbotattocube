import os
import pickle
from typing import List, Dict, Any
from pathlib import Path

import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = Path(__file__).resolve().parent

class Config:
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    LLM_MODEL = "meta-llama/llama-3.3-8b-instruct:free"

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    JSON_FAISS_INDEX = BASE_DIR / "models" / "embeddings.faiss"
    JSON_METADATA = BASE_DIR / "models" / "metadata.pkl"
    DOC_FAISS_INDEX = BASE_DIR / "models" / "word_embeddings.faiss"
    DOC_METADATA = BASE_DIR / "models" / "word_metadata.pkl"

    RETRIEVAL_TOP_K = 3
    HISTORY_MAX_TURNS = 5
    HTTP_REFERER = os.environ.get("HTTP_REFERER", "http://localhost:5000")
    APP_TITLE = os.environ.get("APP_TITLE", "Attocube Assistant")

# ==============================
# RAG PIPELINE
# ==============================
class RAGPipeline:
    def __init__(self, config: Config):
        self.config = config
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set.")

        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        # Load JSON embeddings
        try:
            self.json_index = faiss.read_index(str(config.JSON_FAISS_INDEX))
            with open(config.JSON_METADATA, "rb") as f:
                self.json_metadata = pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError("JSON embeddings missing.")

        # Load DOC embeddings
        try:
            self.doc_index = faiss.read_index(str(config.DOC_FAISS_INDEX))
            with open(config.DOC_METADATA, "rb") as f:
                self.doc_metadata = pickle.load(f)
        except FileNotFoundError:
            print("DOC embeddings missing. Only JSON will be used.")
            self.doc_index, self.doc_metadata = None, []

        # BM25 for JSON and DOC
        self.json_bm25 = BM25Okapi([d["text"].split() for d in self.json_metadata])
        if self.doc_metadata:
            self.doc_bm25 = BM25Okapi([d["text"].split() for d in self.doc_metadata])
        else:
            self.doc_bm25 = None

        # OpenRouter client
        self.client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)

        # Prompt template
        self.prompt_template = """
You are **Attocube Assistant**, an expert helper for Attocube products.
Answer using **only** the provided Knowledge Base snippets.

### RULES
- Start with a **bold one-sentence summary**.
- Keep answers **short, clear, and factual**.
- Use **bullet points** for specs, lists, or comparisons.
- Highlight key terms in **bold**.
- If the answer is not found, reply: **"Not available in knowledge base."**

### Knowledge Base Snippets:
{context}

### User Query: "{query}"
"""

    # --- Intent detection
    def detect_intent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["controller", "amc", "anc"]):
            return "controller"
        elif any(k in q for k in ["ecs", "x", "stage", "model"]):
            return "model"
        else:
            return "general"

    # --- Retrieve chunks
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        results = []

        # JSON FAISS
        dists, idxs = self.json_index.search(vec, self.config.RETRIEVAL_TOP_K)
        for i, idx in enumerate(idxs[0]):
            if idx == -1: continue
            meta = self.json_metadata[idx]
            results.append({"text": meta["text"], "model": meta.get("model", "JSON"), "id": meta.get("id", "JSON"), "score": round(float(dists[0][i]),3)})

        # JSON BM25 fallback
        bm25_scores = self.json_bm25.get_scores(query.split())
        for idx in np.argsort(bm25_scores)[::-1][:self.config.RETRIEVAL_TOP_K]:
            meta = self.json_metadata[idx]
            if meta["id"] not in [r["id"] for r in results]:
                results.append({"text": meta["text"], "model": meta.get("model","JSON"), "id": meta.get("id","JSON"), "score": round(float(bm25_scores[idx]),3)})

        # DOC retrieval
        if self.doc_index:
            dists, idxs = self.doc_index.search(vec, self.config.RETRIEVAL_TOP_K)
            for i, idx in enumerate(idxs[0]):
                if idx == -1: continue
                meta = self.doc_metadata[idx]
                results.append({"text": meta["text"], "model":"DOC","id":"DOC","score": round(float(dists[0][i]),3)})
            # DOC BM25
            bm25_scores = self.doc_bm25.get_scores(query.split())
            for idx in np.argsort(bm25_scores)[::-1][:self.config.RETRIEVAL_TOP_K]:
                meta = self.doc_metadata[idx]
                if meta["id"] not in [r["id"] for r in results]:
                    results.append({"text": meta["text"], "model":"DOC","id":"DOC","score": round(float(bm25_scores[idx]),3)})

        # Sort by score
        return sorted(results, key=lambda x: x["score"], reverse=True)[:self.config.RETRIEVAL_TOP_K]

    # --- LLM query
    def query_llm(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        context = "\n".join([f"- {r['text']}" for r in chunks]) or "No relevant info found."
        prompt = self.prompt_template.format(context=context, query=query)
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[{"role":"user","content":prompt}],
                extra_headers={"HTTP-Referer": self.config.HTTP_REFERER, "X-Title": self.config.APP_TITLE}
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return "⚠️ Error: LLM request failed."

    # --- Generate response
    def generate_response(self, query: str) -> Dict[str,str]:
        intent = self.detect_intent(query)
        chunks = self.retrieve(query)
        answer = self.query_llm(query, chunks)
        sources = "\n".join([f"- **{r['model']}** (ID:{r['id']}, score:{r['score']})" for r in chunks])
        return {"response": f"{answer}\n\n📌 **Sources:**\n{sources}", "intent": intent}

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)
config = Config()
rag_pipeline = RAGPipeline(config)

@app.route("/")
def home():
    return render_template("index.html", config=config)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("user_input")
    if not user_input:
        return jsonify({"error":"No user_input provided"}), 400
    result = rag_pipeline.generate_response(user_input)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
