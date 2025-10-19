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

# === 3. Load forecast terbaru ===
try:
    df_forecast = pd.read_excel("ihsg_forecast.xlsx")
except FileNotFoundError:
    raise SystemExit("❌ File 'ihsg_forecast.xlsx' tidak ditemukan!")

# Bersihkan dan format data
numeric_cols = ["Terakhir", "Pembukaan", "Tertinggi", "Terendah"]
for col in numeric_cols:
    if col in df_forecast.columns:
        df_forecast[col] = pd.to_numeric(df_forecast[col], errors="coerce").round(2)

df_forecast = df_forecast.replace([np.inf, -np.inf], np.nan)
df_forecast = df_forecast.where(pd.notnull(df_forecast), None)

df_forecast["Tanggal"] = pd.to_datetime(df_forecast["Tanggal"]).dt.strftime("%m/%d/%Y")

# === 4. Ambil Sheet1 ===
sheet1 = client.open_by_key(SPREADSHEET_ID).worksheet("Sheet1")

# Ambil semua data sheet
sheet_values = sheet1.get_all_values()
headers = sheet_values[0]
records = sheet_values[1:]

if "Tanggal" not in headers:
    raise SystemExit("❌ Kolom 'Tanggal' tidak ditemukan di Sheet1.")

tanggal_idx = headers.index("Tanggal")
status_idx = headers.index("Status") if "Status" in headers else None

# === 5. Cari baris dengan tanggal kemarin ===
yesterday = today - datetime.timedelta(days=1)
yesterday_str = yesterday.strftime("%m/%d/%Y")

found_row_index = None
for i, row in enumerate(records, start=2):  # mulai dari baris ke-2 (karena header di baris 1)
    if row[tanggal_idx].strip() == yesterday_str:
        found_row_index = i
        break

if found_row_index:
    print(f"🕐 Update data historis untuk {yesterday_str} di baris {found_row_index}")

    # ambil baris dari df_forecast dengan tanggal yg sama
    updated_row = df_forecast[df_forecast["Tanggal"] == yesterday_str]
    if not updated_row.empty:
        row_data = updated_row.iloc[0].to_dict()

        # isi kolom historis ke sheet
        update_dict = {}
        for key, val in row_data.items():
            if key in headers:
                col_idx = headers.index(key) + 1
                value = "" if val is None or str(val).lower() == "nan" else val
                update_dict[col_idx] = value

        # set kolom Status ke Historis
        if status_idx is not None:
            update_dict[status_idx + 1] = "Historis"

        # update langsung ke Sheet
        for col_idx, value in update_dict.items():
            sheet1.update_cell(found_row_index, col_idx, value)

else:
    print(f"⚠️ Tidak ditemukan baris dengan tanggal {yesterday_str} di Sheet1.")

# === 6. Tambahkan baris prediksi baru (lebih dari tanggal terakhir di Sheet) ===
existing_records = sheet1.get_all_records()
existing_dates = []
for row in existing_records:
    if "Tanggal" in row and row["Tanggal"]:
        try:
            existing_dates.append(pd.to_datetime(row["Tanggal"]))
        except Exception:
            pass
last_date_in_sheet = max(existing_dates) if existing_dates else None

rows_to_add = []
for _, row in df_forecast.iterrows():
    tgl = pd.to_datetime(row["Tanggal"])
    if last_date_in_sheet is None or tgl > last_date_in_sheet:
        cleaned = []
        for val in row.tolist():
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                cleaned.append("")
            else:
                cleaned.append(val)
        rows_to_add.append(cleaned)

if rows_to_add:
    sheet1.append_rows(rows_to_add, value_input_option="USER_ENTERED")
    print(f"✅ {len(rows_to_add)} baris prediksi baru ditambahkan.")
else:
    print("ℹ️ Tidak ada baris prediksi baru untuk ditambahkan.")

# === 7. Upload evaluasi model ===
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

print("🎉 Upload selesai.")