"""
Router — menentukan apakah query user butuh RAG (informasi/kebijakan)
atau tool call (aksi spesifik, misal cek status tiket).

Pola ini mengikuti workflow router yang sudah dipelajari di Sesi 12.4:
  user input -> intent detection -> jika butuh data -> retrieval (RAG)
                                  -> jika butuh aksi -> tool call
"""

from app.rag import answer_from_knowledge_base
from app.tools import get_ticket_status

# 🔧 TODO: import LLM client di sini (Gemini AI API) untuk melakukan intent detection
import logging
import re
 
from google.genai import types
 
from app.rag import LLM_MODEL, answer_from_knowledge_base, get_client
from app.tools import get_ticket_status
 
logger = logging.getLogger(__name__)
 
TICKET_ID_PATTERN = re.compile(r"\bTICKET[\s_-]*(\d{1,4})\b", re.IGNORECASE)

# Kata status/cek berdekatan dengan kata tiket (urutan bebas)
_STATUS_WORDS = r"(status|cek|check|progres|progress|perkembangan|sampai mana|sudah selesai|update)"
_TICKET_WORDS = r"(tiket|ticket)"
TICKET_STATUS_PATTERN = re.compile(
    rf"{_STATUS_WORDS}.{{0,40}}{_TICKET_WORDS}|{_TICKET_WORDS}.{{0,40}}{_STATUS_WORDS}",
    re.IGNORECASE,
)
TICKET_MENTION_PATTERN = re.compile(_TICKET_WORDS, re.IGNORECASE)
 
 
def _rule_based_intent(query: str) -> str | None:
    """Baseline keyword/regex. Return None jika ambigu."""
    if TICKET_ID_PATTERN.search(query) or TICKET_STATUS_PATTERN.search(query):
        return "ticket_status"
    if not TICKET_MENTION_PATTERN.search(query):
        return "faq"
    return None  # menyebut 'tiket' tetapi tidak jelas -> serahkan ke LLM
 
async def _llm_intent(query: str) -> str:
    prompt = (
        "Klasifikasikan pertanyaan berikut ke dalam salah satu kategori:\n"
        "'faq' (pertanyaan informasi/kebijakan umum) atau\n"
        "'ticket_status' (permintaan mengecek status tiket tertentu).\n"
        "Jawab hanya dengan satu kata.\n"
        f"Pertanyaan: {query}"
    )
    response = await get_client().aio.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    label = (response.text or "").strip().lower()
    return "ticket_status" if "ticket_status" in label else "faq"
 


async def detect_intent(query: str) -> str:
    """
    Menentukan intent dari query user. Harus mengembalikan salah satu:
      - "faq"           -> pertanyaan informasi/kebijakan, arahkan ke RAG
      - "ticket_status"  -> permintaan cek status tiket, arahkan ke tool call

    # TODO: implementasikan intent detection menggunakan LLM (structured prompt,
    # lihat Sesi 10.1) ATAU pendekatan sederhana berbasis keyword sebagai baseline
    # sebelum beralih ke pendekatan berbasis LLM.
    #
    # Contoh prompt yang bisa dipakai sebagai starting point:
    #
    #   "Klasifikasikan pertanyaan berikut ke dalam salah satu kategori:
    #    'faq' (pertanyaan informasi/kebijakan umum) atau
    #    'ticket_status' (permintaan mengecek status tiket tertentu).
    #    Jawab hanya dengan satu kata.
    #    Pertanyaan: {query}"
    # Return 'faq' atau 'ticket_status'. Rule-based dulu, LLM hanya untuk kasus ambigu.
    """
    intent = _rule_based_intent(query)
    if intent is not None:
        return intent
    try:
        return await _llm_intent(query)
    except Exception:
        logger.warning("Intent detection via LLM gagal, fallback ke 'faq'", exc_info=True)
        return "faq"

def extract_ticket_id(query: str) -> str | None:
    """Ekstrak ID tiket dan normalisasi ke format TICKET-XXXX (mis. 'ticket 42' -> TICKET-0042)."""
    match = TICKET_ID_PATTERN.search(query)
    if not match:
        return None
    return f"TICKET-{int(match.group(1)):04d}"

async def handle_query(query: str) -> dict:
    """
    Fungsi utama yang dipanggil oleh endpoint /ask.
    Menjalankan intent detection, lalu merutekan ke RAG atau tool call yang sesuai.
    """
    intent = await detect_intent(query)
 
    if intent == "ticket_status":
        ticket_id = extract_ticket_id(query)
        if ticket_id is None:
            return {
                "status": "error",
                "intent": "ticket_status",
                "data": "Mohon sebutkan ID tiket Anda, contoh: TICKET-0042.",
            }
        data = get_ticket_status(ticket_id)
        return {"status": "success", "intent": "ticket_status", "data": data}
 
    if intent == "faq":
        data = await answer_from_knowledge_base(query)
        return {"status": "success", "intent": "faq", "data": data}
 
    return {"status": "error", "intent": "unknown", "data": "Maaf, saya tidak memahami permintaan Anda."}
