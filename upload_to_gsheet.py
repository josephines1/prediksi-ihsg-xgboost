"""
upload_to_gsheet.py
"""

import os
import json
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np
import datetime
import holidays
import sys

# ===============================
# CEK LIBUR NASIONAL & AKHIR PEKAN
# ===============================
today = datetime.date.today()
id_holidays = holidays.Indonesia()

if today.weekday() >= 5 or today in id_holidays:
    reason = "akhir pekan" if today.weekday() >= 5 else "libur nasional"
    print(f"⏸️ Hari ini ({today}) adalah {reason}. Prediksi otomatis dilewati.")
    sys.exit(0)

# === 1. Load kredensial ===
creds_json = os.environ.get("GOOGLE_CREDENTIALS")

if creds_json:
    print("🔐 Menggunakan kredensial dari environment (GitHub Secrets)...")
    creds_dict = json.loads(creds_json)
else:
    print("🧩 Menggunakan kredensial lokal...")
    with open("prediksi-ihsg-xgboost-fe9d150cc374.json") as f:
        creds_dict = json.load(f)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
client = gspread.authorize(creds)

# === 2. Konfigurasi Spreadsheet ===
SPREADSHEET_ID = "13EfYovEBQHg19pDY21I8RV7RR4Mo1SNSRzNckg_dwhw"

# === 3. Upload file prediksi utama (Sheet1) ===
try:
    df_forecast = pd.read_excel("ihsg_forecast.xlsx")
except FileNotFoundError:
    raise SystemExit("❌ File 'ihsg_forecast.xlsx' tidak ditemukan!")

# Bersihkan nilai yang tidak valid
df_forecast = df_forecast.replace([np.inf, -np.inf], np.nan).fillna("")

# Kolom yang seharusnya numerik
numeric_cols = ["Terakhir", "Pembukaan", "Tertinggi", "Terendah"]

# Konversi dulu ke float, lalu bulatkan dan ubah ke format lokal (koma desimal)
for col in numeric_cols:
    if col in df_forecast.columns:
        df_forecast[col] = (
            pd.to_numeric(df_forecast[col], errors="coerce")
            .round(2)
            .map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "")
        )

# Kolom lain diubah jadi string agar tidak error saat upload
for col in df_forecast.columns:
    if col not in numeric_cols:
        df_forecast[col] = df_forecast[col].astype(str)

# Ambil atau buat Sheet1
try:
    sheet1 = client.open_by_key(SPREADSHEET_ID).worksheet("Sheet1")
except gspread.WorksheetNotFound:
    client.open_by_key(SPREADSHEET_ID).add_worksheet(title="Sheet1", rows="1000", cols="20")
    sheet1 = client.open_by_key(SPREADSHEET_ID).worksheet("Sheet1")

# Tulis ulang data
sheet1.clear()
sheet1.update([df_forecast.columns.values.tolist()] + df_forecast.values.tolist())
print("✅ Data prediksi berhasil diupload ke Sheet1")

# === 4. Upload evaluasi model (Sheet2) kalau ada ===
if os.path.exists("model_evaluation.xlsx"):
    df_eval = pd.read_excel("model_evaluation.xlsx", engine="openpyxl")
    df_eval = df_eval.replace([np.inf, -np.inf], np.nan).fillna("")
    df_eval = df_eval.astype(str)

    try:
        sheet2 = client.open_by_key(SPREADSHEET_ID).worksheet("Sheet2")
    except gspread.WorksheetNotFound:
        client.open_by_key(SPREADSHEET_ID).add_worksheet(title="Sheet2", rows="100", cols="10")
        sheet2 = client.open_by_key(SPREADSHEET_ID).worksheet("Sheet2")

    sheet2.clear()
    sheet2.update([df_eval.columns.values.tolist()] + df_eval.values.tolist())
    print("📊 Evaluasi model berhasil diupload ke Sheet2")
else:
    print("ℹ️ File evaluasi model belum ada, skip upload Sheet2")

print("🎉 Semua data berhasil diupload ke Google Sheets!")