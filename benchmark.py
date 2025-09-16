import os
import pickle
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- Configuration ---
class Config:
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    JSON_FAISS_INDEX = r"C:\Users\mayan\Desktop\newbot\models\embeddings.faiss"
    JSON_METADATA = r"C:\Users\mayan\Desktop\newbot\models\metadata.pkl"

    # Free models to test on OpenRouter
    MODELS = [
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-nemo:free",
        "meta-llama/llama-3-8b-instruct:free"
    ]

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    HTTP_REFERER = "http://localhost"
    APP_TITLE = "BenchmarkBot"

    RETRIEVAL_TOP_K = 3


# --- RAG Pipeline ---
class RAGPipeline:
    def __init__(self, config: Config):
        self.config = config

        # Load embedding model + FAISS index
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index = faiss.read_index(config.JSON_FAISS_INDEX)
        with open(config.JSON_METADATA, "rb") as f:
            self.kb_metadata = pickle.load(f)

        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set. Please export it.")

        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY
        )

        self.prompt_template = """
You are a helpful AI assistant for Attocube products.
Answer the question based ONLY on the 'Knowledge Base Snippets' below.
Be concise, use **bold** for key terms, and provide a short, clear summary.

---
### Knowledge Base Snippets:
{context}
---
### User Query: "{query}"
"""

    def retrieve_chunks(self, query: str) -> List[Dict]:
        vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(vec, self.config.RETRIEVAL_TOP_K)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            metadata_item = self.kb_metadata[idx]
            results.append({
                "text": metadata_item.get("text", ""),
                "model": metadata_item.get("model", "N/A"),
                "id": metadata_item.get("id", "N/A"),
                "score": round(float(distances[0][i]), 3)
            })
        return results

    def query_model(self, model: str, query: str, context: str) -> str:
        prompt = self.prompt_template.format(context=context, query=query)
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={
                    "HTTP-Referer": self.config.HTTP_REFERER,
                    "X-Title": self.config.APP_TITLE
                }
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error querying {model}: {e}]"


# --- Benchmark Runner ---
def run_benchmark():
    config = Config()
    rag = RAGPipeline(config)

    test_queries = [
        "What is ECS?",
        "Which controllers are used with ECS series?",
        "Compare AMC100 and AMC300.",
        "What is the footprint of ECSx3030/AI/RT?"
    ]

    for q in test_queries:
        print("=" * 80)
        print(f"🔎 Query: {q}\n")

        # Retrieve chunks for context
        chunks = rag.retrieve_chunks(q)
        context = "\n".join([f"- {c['text']}" for c in chunks])

        for model in config.MODELS:
            print(f"--- Model: {model} ---")
            answer = rag.query_model(model, q, context)
            print(answer)
            print("\n")


if __name__ == "__main__":
    run_benchmark()
