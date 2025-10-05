import os
import json
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np

# === Cek apakah sedang di GitHub Actions atau lokal ===
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

# === Konfigurasi ===
SPREADSHEET_ID = "13EfYovEBQHg19pDY21I8RV7RR4Mo1SNSRzNckg_dwhw"  # ganti dengan ID Sheet kamu
SHEET_NAME = "Sheet1"  # ganti sesuai nama tab di Google Sheet kamu

# === Autentikasi ===
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

# === Baca dan bersihkan data ===
df = pd.read_excel("ihsg_forecast.xlsx")

# Hapus nilai NaN, inf, -inf
df = df.replace([np.inf, -np.inf], np.nan).fillna("")

# Konversi semua nilai ke string (supaya JSON valid)
df = df.astype(str)

# === Kosongkan isi lama dan tulis data baru ===
sheet.clear()
sheet.update([df.columns.values.tolist()] + df.values.tolist())

print("✅ Data berhasil diupload ke Google Sheets!")
