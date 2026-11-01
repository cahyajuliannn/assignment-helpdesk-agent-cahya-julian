# IT Helpdesk Assistant (RAG + Mini-Agent Router)

Backend API berbasis FastAPI yang menggabungkan **RAG** (menjawab pertanyaan kebijakan/informasi dari knowledge base) dan **router/tool call** (mengecek status tiket) dalam satu endpoint `/ask`, dikemas dalam Docker.

## 1. Problem Statement

**Objective.** Tim People & IT Operations menerima banyak pertanyaan repetitif dari karyawan. Sistem ini mengotomatiskan dua jenis permintaan dalam satu asisten AI.

| Jenis permintaan | Contoh | Ditangani oleh |
|---|---|---|
| Informasi / kebijakan (`faq`) | "Berapa jatah cuti tahunan saya?", "Bagaimana cara reset password VPN?" | RAG (FAISS + Gemini) |
| Cek data spesifik (`ticket_status`) | "Apa status tiket TICKET-0005?" | Tool call ke `mock_tickets.csv` |

**Asumsi & batasan**
- Knowledge base (31 FAQ) dan data tiket (15 tiket) adalah data mock dari starter repository.
- Tool hanya membaca status tiket (tidak membuat/mengubah tiket).
- Tidak ada autentikasi user; siapa pun bisa mengecek tiket manapun.
- Aplikasi dijalankan lokal (tanpa deploy cloud).

## 2. Desain Workflow

```
query masuk (POST /ask)
        │
        ▼
 intent detection (router.py)
   1. regex/keyword  ──► jelas? ──┐
   2. LLM (hanya jika ambigu) ────┘
        │
   ┌────┴─────────────┐
   ▼                  ▼
 faq               ticket_status
 rag.py             extract_ticket_id → tools.py
 embed query →      baca mock_tickets.csv
 FAISS top-3 →      (tidak ketemu → pesan informatif)
 prompt P/T/C/F →
 Gemini
   └────────┬─────────┘
            ▼
 {"status", "intent", "data"}
```

**Kriteria intent**
- `ticket_status`: query memuat ID tiket (`TICKET-XXXX`, termasuk variasi "ticket 42"), **atau** kata cek/status/progres berdekatan dengan kata "tiket/ticket".
- `faq`: query yang tidak menyebut tiket, atau pertanyaan prosedur seperti "cara mengajukan laptop lewat tiket IT".
- Query yang menyebut "tiket" tetapi tidak cocok pola di atas dianggap ambigu dan diklasifikasi LLM (fallback ke `faq` jika LLM gagal).

**Keputusan desain lain**
- Index FAISS (`IndexFlatIP` + normalisasi L2 = cosine similarity) dibangun sekali saat startup. Jika gagal, dibangun lazy saat request pertama.
- Unit retrieval = satu FAQ (kategori + pertanyaan + jawaban).
- Jika skor retrieval teratas < `MIN_RELEVANCE_SCORE` (default 0.55), sistem menjawab "tidak ada informasi" tanpa memanggil LLM. Ini mencegah halusinasi pada pertanyaan di luar cakupan.
- Error handling: LLM gagal → HTTP 503 dengan body terstruktur; query kosong → 400; error tak terduga → 500; tiket tidak ditemukan → pesan informatif (tidak crash).

## 3. Menjalankan Aplikasi

Prasyarat: Python 3.11+, API key Gemini.

```bash
cp .env.example .env        # lalu isi GEMINI_API_KEY
```

### Tanpa Docker
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Dengan Docker
```bash
docker build --no-cache -t helpdesk-agent .
docker run --rm -p 8000:8000 --env-file .env helpdesk-agent
```
API key tidak masuk ke image (`.env` ada di `.dockerignore`), hanya disuplai saat runtime.

### Mencoba
Swagger UI: http://localhost:8000/docs
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "Apa status tiket TICKET-0005?"}'
```

## 4. Hasil Pengujian 5 Skenario

> Isi kolom "Respons aktual" dengan output sebenarnya dari `/ask` setelah kamu menjalankan aplikasi.

| # | Query | Intent diharapkan | Intent aktual | Respons aktual | Router tepat? |
|---|---|---|---|---|---|
| 1 | Berapa jatah cuti tahunan karyawan tetap? | faq | | | |
| 2 | Bagaimana cara reset password VPN? | faq | | | |
| 3 | Apa status tiket TICKET-0005 saya? | ticket_status | | | |
| 4 | Tolong cek status TICKET-9999 | ticket_status (tiket tidak ada) | | | |
| 5 | Bagaimana cara membuat nasi goreng? (di luar cakupan) | faq → fallback "tidak ada info" | | | |

**Observasi:** _(tulis di sini: apakah routing sesuai, kualitas jawaban RAG, perilaku pada tiket tak ditemukan dan query di luar cakupan.)_

## 5. Keterbatasan
- Intent detection berbasis regex/keyword rapuh terhadap parafrase tak biasa; kasus ambigu bergantung pada LLM.
- Threshold relevansi (`MIN_RELEVANCE_SCORE`) bersifat heuristik dan perlu dikalibrasi untuk model embedding yang dipakai.
- Query campuran (misal: tanya kebijakan sekaligus cek tiket) hanya diarahkan ke satu intent.
- Tidak ada autentikasi/otorisasi pada data tiket, dan data tiket dibaca dari CSV statis.
- Satu knowledge base kecil (31 FAQ); tidak ada re-ranking, chunking, maupun memori percakapan.
- Index FAISS disimpan di memori dan dibangun ulang tiap container start (membutuhkan panggilan embedding).