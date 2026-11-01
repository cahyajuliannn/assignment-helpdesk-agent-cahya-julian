"""
Modul Tools — fungsi-fungsi "aksi" yang dipanggil router ketika intent
BUKAN pertanyaan informasi/kebijakan, melainkan permintaan spesifik
yang butuh pengecekan data langsung (bukan retrieval semantik).

Pola ini mengikuti konsep Tool Integration di Sesi 12.3.
"""

import logging
import os
 
import pandas as pd
 
logger = logging.getLogger(__name__)
 
MOCK_TICKETS_PATH = os.getenv("MOCK_TICKETS_PATH", "data/mock_tickets.csv")
 
_tickets_df = None

def _load_tickets() -> pd.DataFrame:
    global _tickets_df
    if _tickets_df is None:
        _tickets_df = pd.read_csv(MOCK_TICKETS_PATH, dtype=str)
    return _tickets_df


def get_ticket_status(ticket_id: str) -> str:
    """
    Mencari status tiket berdasarkan ticket_id pada data mock_tickets.csv.

    # TODO: implementasikan pencarian data tiket berdasarkan ticket_id.
    # Jika ticket_id tidak ditemukan, kembalikan pesan yang informatif
    # (bukan error yang membuat aplikasi crash).
    #
    # Contoh langkah:
    #   1. Muat mock_tickets.csv menggunakan pandas
    #   2. Cari baris dengan ticket_id yang cocok
    #   3. Jika ditemukan, susun string jawaban (subject, status, last_update)
    #   4. Jika tidak ditemukan, kembalikan pesan bahwa tiket tidak ditemukan
    """
    if not ticket_id:
        return "ID tiket tidak ditemukan pada permintaan Anda. Sebutkan ID tiket, contoh: TICKET-0042."
 
    try:
        df = _load_tickets()
    except Exception:
        logger.exception("Gagal memuat data tiket")
        return "Maaf, data tiket sedang tidak dapat diakses. Silakan coba lagi nanti."
 
    match = df[df["ticket_id"].str.upper() == ticket_id.strip().upper()]
    if match.empty:
        return f"Tiket {ticket_id} tidak ditemukan. Mohon periksa kembali ID tiket Anda."
 
    row = match.iloc[0]
    return (
        f"Tiket {row['ticket_id']} ({row['category']}): \"{row['subject']}\" "
        f"- Status: {row['status']}, terakhir diperbarui {row['last_update']}."
    )
