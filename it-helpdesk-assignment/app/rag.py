"""
Modul RAG — mengubah knowledge base menjadi embedding, menyimpannya ke FAISS,
melakukan semantic search, lalu menghasilkan jawaban menggunakan LLM.

Alur ini serupa dengan yang sudah dibangun di Assignment 3 (Semantic Search &
LLM Application) dan Sesi 11 (Building a Simple RAG System) — di assignment ini,
alur tersebut dibungkus menjadi fungsi yang dipanggil oleh router.py.
"""

import asyncio
import logging
import os
import threading
 
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/it_helpdesk_knowledge_base.csv")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Skor cosine minimum agar hasil retrieval dianggap relevan (kalibrasi bila perlu).
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.55"))

FALLBACK_MESSAGE = (
    "Maaf, saya tidak menemukan informasi yang relevan di knowledge base. "
    "Silakan hubungi tim People/IT Support atau buat tiket IT untuk bantuan lebih lanjut."
)

class LLMError(Exception):
    """Dilempar ketika Gemini API gagal merespons / API key tidak tersedia."""

# 🔧 TODO: muat knowledge base dan bangun index FAISS saat aplikasi startup
# (bukan di setiap request), agar lebih efisien. Pertimbangkan menggunakan
# FastAPI startup event atau lazy-loading dengan cache sederhana.

_client = None
_index = None
_df = None
_lock = threading.Lock()

def get_client():
    """Client dibuat lazy agar aplikasi tetap bisa start walau API key belum diset."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY belum diset di environment.")
        _client = genai.Client(api_key=api_key)
    return _client

def _embed(texts: list[str], task_type: str) -> np.ndarray:
    try:
        result = get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Gagal membuat embedding: {e}") from e
 
    vectors = np.array([e.values for e in result.embeddings], dtype="float32")
    faiss.normalize_L2(vectors)  # cosine similarity = inner product pada vektor ternormalisasi
    return vectors

def build_index():
    """Bangun index FAISS dari knowledge base (idempotent, dipanggil sekali saat startup)."""
    global _index, _df
    with _lock:
        if _index is not None:
            return
        df = pd.read_csv(KNOWLEDGE_BASE_PATH)
        # Unit retrievable: satu FAQ = pertanyaan + jawaban (+ kategori)
        df["document"] = (
            "Kategori: " + df["category"].astype(str)
            + "\nPertanyaan: " + df["question"].astype(str)
            + "\nJawaban: " + df["answer"].astype(str)
        )
        vectors = _embed(df["document"].tolist(), "RETRIEVAL_DOCUMENT")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        _df, _index = df, index
        logger.info("Index FAISS dibangun: %d dokumen", len(df))

def semantic_search(query: str, top_k: int = 3) -> list[dict]:
    """Cari top_k FAQ paling relevan. Mengembalikan list dict terurut dari skor tertinggi."""
    if _index is None:
        build_index()
    query_vec = _embed([query], "RETRIEVAL_QUERY")
    scores, ids = _index.search(query_vec, min(top_k, len(_df)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        row = _df.iloc[int(idx)]
        results.append({
            "faq_id": int(row["faq_id"]),
            "category": row["category"],
            "question": row["question"],
            "answer": row["answer"],
            "score": float(score),
        })
    return results

def _get_embedding(text: str):
    result = _client.models.embed_content(
        model="text-embedding-004",  # sesuaikan versi model embedding
        contents=text,
    )
    return result.embeddings[0].values

def build_prompt(query: str, results: list[dict]) -> str:
    """Prompt terstruktur: Persona / Task / Context / Format."""
    context = "\n\n".join(
        f"[FAQ {r['faq_id']} - {r['category']}]\nQ: {r['question']}\nA: {r['answer']}"
        for r in results
    )
    return f"""PERSONA:
Anda adalah asisten helpdesk internal perusahaan (People & IT Operations) yang ramah, jelas, dan ringkas.
 
TASK:
Jawab pertanyaan karyawan HANYA berdasarkan CONTEXT di bawah. Jangan mengarang informasi.
Jika CONTEXT tidak memuat jawaban yang cukup, katakan bahwa Anda tidak memiliki informasinya
dan sarankan menghubungi tim People/IT Support.
 
CONTEXT:
{context}
 
FORMAT:
Jawab dalam Bahasa Indonesia, maksimal 3-4 kalimat, tanpa menyebut kata "CONTEXT" atau nomor FAQ.
 
PERTANYAAN KARYAWAN:
{query}
 
JAWABAN:"""
 

async def answer_from_knowledge_base(query: str) -> str:
    """Dipanggil router ketika intent = 'faq'."""
    results = await asyncio.to_thread(semantic_search, query, 3)
    if not results or results[0]["score"] < MIN_RELEVANCE_SCORE:
        return FALLBACK_MESSAGE
 
    prompt = build_prompt(query, results)
    try:
        response = await get_client().aio.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Gagal memanggil LLM: {e}") from e
 
    text = (response.text or "").strip()
    if not text:
        raise LLMError("LLM mengembalikan respons kosong.")
    return text
