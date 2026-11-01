"""
Entry point aplikasi FastAPI — IT Helpdesk Assistant.

Alur endpoint utama:
  1. Menerima pertanyaan/permintaan dari user
  2. Meneruskan ke router (router.py) untuk menentukan intent
  3. Router memanggil RAG (rag.py) atau tool (tools.py) sesuai intent
  4. Mengembalikan response terstruktur ke client
"""

import asyncio
import logging
from contextlib import asynccontextmanager
 
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
 
from app.rag import LLMError, build_index
from app.router import handle_query
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bangun index FAISS sekali saat startup. Jika gagal (mis. API key belum diset),
    # aplikasi tetap hidup dan index akan dicoba lagi (lazy) saat request FAQ pertama.
    try:
        await asyncio.to_thread(build_index)
    except Exception:
        logger.warning("Index belum dapat dibangun saat startup; akan dicoba saat request.", exc_info=True)
    yield
 
app = FastAPI(title="IT Helpdesk Assistant API", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class AskResponse(BaseModel):
    status: str
    intent: str
    data: str


@app.get("/health")
async def health():
    """Endpoint sederhana untuk memastikan service berjalan."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Endpoint utama. Menerima pertanyaan bebas dari user (bisa berupa
    pertanyaan informasi/kebijakan ATAU permintaan cek status tiket),
    lalu meneruskannya ke router untuk diproses.

    # TODO: tambahkan error handling untuk kondisi LLM gagal merespons
    # atau intent tidak dapat dikenali (lihat panduan error handling di Sesi 13.4)
    """
    query = request.query.strip()
    if not query:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "intent": "unknown", "data": "Query tidak boleh kosong."},
        )
    try:
        result = await handle_query(query)
    except LLMError as e:
        logger.error("LLM error: %s", e)
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "intent": "faq",
                "data": "Layanan AI sedang tidak dapat merespons. Silakan coba lagi beberapa saat.",
            },
        )
    except Exception:
        logger.exception("Unhandled error pada /ask")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "intent": "unknown", "data": "Terjadi kesalahan internal pada server."},
        )
    return AskResponse(**result)
