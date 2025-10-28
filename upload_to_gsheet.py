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

from dotenv import load_dotenv

# ===============================
# Load env
# ===============================
load_dotenv()

# Akses variabel
spreadsheet_id_dev = os.getenv("SPREADSHEET_ID_DEV")
spreadsheet_id_prod = os.getenv("SPREADSHEET_ID_PROD")
env = os.getenv("ENV")

if env == "development":
    SPREADSHEET_ID = spreadsheet_id_dev
else:
    SPREADSHEET_ID = spreadsheet_id_prod

print(f"🚀 Environment aktif: {env}")
print(f"🧾 Spreadsheet ID yang digunakan: {SPREADSHEET_ID}")

# ===============================
# CEK LIBUR NASIONAL & AKHIR PEKAN
# ===============================
today = datetime.date.today()
id_holidays = holidays.Indonesia()

def is_trading_day(date):
    """Cek apakah tanggal adalah hari bursa (bukan weekend/libur)"""
    return date.weekday() < 5 and date not in id_holidays

if today.weekday() >= 5 or today in id_holidays:
    reason = "akhir pekan" if today.weekday() >= 5 else "libur nasional"
    print(f"⏸️ Hari ini ({today}) adalah {reason}. Prediksi otomatis dilewati.")
    sys.exit(0)

print(f"📅 Hari ini: {today.strftime('%d/%m/%Y (%A)')}")

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

# === 2. Load forecast terbaru ===
try:
    df_forecast = pd.read_excel("ihsg_forecast.xlsx")
    print(f"📊 File ihsg_forecast.xlsx berisi {len(df_forecast)} baris")
except FileNotFoundError:
    raise SystemExit("❌ File 'ihsg_forecast.xlsx' tidak ditemukan!")

# Bersihkan dan format data
numeric_cols = ["Terakhir", "Pembukaan", "Tertinggi", "Terendah", "Terakhir (Prediksi)"]
for col in numeric_cols:
    if col in df_forecast.columns:
        df_forecast[col] = pd.to_numeric(df_forecast[col], errors="coerce").round(2)

df_forecast = df_forecast.replace([np.inf, -np.inf], np.nan)
df_forecast = df_forecast.where(pd.notnull(df_forecast), None)

# Parse tanggal jadi datetime object dulu
df_forecast["Tanggal"] = pd.to_datetime(df_forecast["Tanggal"])

# === 3. Ambil Sheet1 ===
sheet1 = client.open_by_key(SPREADSHEET_ID).worksheet("Sheet1")

# Ambil semua data sheet
sheet_values = sheet1.get_all_values()
headers = sheet_values[0]
records = sheet_values[1:]

if "Tanggal" not in headers:
    raise SystemExit("❌ Kolom 'Tanggal' tidak ditemukan di Sheet1.")

tanggal_idx = headers.index("Tanggal")
terakhir_prediksi_idx = headers.index("Terakhir (Prediksi)") if "Terakhir (Prediksi)" in headers else None

print(f"📋 Sheet1 memiliki {len(records)} baris data")

# === 4. Update baris yang ada: data historis & prediksi ===
print("🔄 Mengecek dan mengupdate baris yang sudah ada...")

all_cells_to_update = []

def normalize_numeric(val):
    """Konversi string dengan format lokal (ID/EN) menjadi float untuk perbandingan."""
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    val = str(val).strip()

    # Hapus simbol %, tanda mata uang, dan spasi
    val = val.replace("%", "").replace("Rp", "").strip()

    # Deteksi pola numerik
    try:
        # Kalau format Indonesia (contoh: 8.274,35)
        if "," in val and "." in val and val.find(",") > val.find("."):
            val = val.replace(".", "").replace(",", ".")
        # Kalau format Inggris (contoh: 8,274.35)
        elif "," in val and "." in val and val.find(",") < val.find("."):
            val = val.replace(",", "")
        # Kalau hanya koma sebagai desimal
        elif "," in val and "." not in val:
            val = val.replace(",", ".")
        return round(float(val), 2)
    except Exception:
        return None

for i, row in enumerate(records, start=2):
    if len(row) <= tanggal_idx or not row[tanggal_idx].strip():
        continue
    
    row_date_str = row[tanggal_idx].strip()
    try:
        sheet_date = pd.to_datetime(row_date_str, format="%d/%m/%Y")
    except Exception:
        continue

    forecast_row = df_forecast.loc[df_forecast["Tanggal"].dt.date == sheet_date.date()]
    if forecast_row.empty:
        continue
    forecast_row = forecast_row.iloc[0]

    col_mapping_hist = {
        "Terakhir": "Terakhir",
        "Pembukaan": "Pembukaan",
        "Tertinggi": "Tertinggi",
        "Terendah": "Terendah",
        "Vol.": "Vol.",
        "Perubahan%": "Perubahan%"
    }

    col_mapping_pred = {"Terakhir (Prediksi)": "Terakhir (Prediksi)"}

    terakhir_prediksi_val = ""
    if terakhir_prediksi_idx is not None and len(row) > terakhir_prediksi_idx:
        terakhir_prediksi_val = row[terakhir_prediksi_idx].strip()

    # CASE 1 — update data historis (tanggal < hari ini)
    if sheet_date.date() < today:
        for col_name, df_col in col_mapping_hist.items():
            if col_name not in headers:
                continue

            col_idx = headers.index(col_name) + 1
            new_val = forecast_row.get(df_col, None)
            current_val = row[col_idx - 1].strip() if len(row) > col_idx - 1 else ""

            # Bandingkan nilai numerik
            new_num = normalize_numeric(new_val)
            old_num = normalize_numeric(current_val)

            # Update kalau kosong atau berbeda signifikan (>0.01)
            if (old_num is None and new_num is not None) or (
                isinstance(new_num, (int, float))
                and isinstance(old_num, (int, float))
                and abs(new_num - old_num) > 0.01
            ):
                col_letter = gspread.utils.rowcol_to_a1(i, col_idx)
                all_cells_to_update.append({'range': col_letter, 'values': [[new_val]]})
                print(f"   🔧 Update {sheet_date.strftime('%d/%m/%Y')} kolom {col_name} → {new_val}")

    # CASE 2 — update prediksi (tanggal >= hari ini)
    elif sheet_date.date() >= today:
        if "Terakhir (Prediksi)" in headers:
            col_idx = headers.index("Terakhir (Prediksi)") + 1
            new_val = forecast_row.get("Terakhir (Prediksi)", None)

            # Hanya update kalau ada nilai baru (bukan NaN)
            if pd.notnull(new_val):
                current_val = row[col_idx - 1].strip() if len(row) > col_idx - 1 else ""
                new_num = normalize_numeric(new_val)
                old_num = normalize_numeric(current_val)

                # Update kalau kosong ATAU nilainya berubah signifikan
                if (old_num is None and new_num is not None) or (
                    isinstance(new_num, (int, float))
                    and isinstance(old_num, (int, float))
                    and abs(new_num - old_num) > 0.01
                ):
                    col_letter = gspread.utils.rowcol_to_a1(i, col_idx)
                    all_cells_to_update.append({'range': col_letter, 'values': [[new_val]]})
                    print(f"   🔮 Update prediksi {sheet_date.strftime('%d/%m/%Y')} → {new_val}")

# Batch update
if all_cells_to_update:
    print(f"📝 Mengupdate {len(all_cells_to_update)} cell...")
    batch_size = 50
    for j in range(0, len(all_cells_to_update), batch_size):
        batch = all_cells_to_update[j:j+batch_size]
        sheet1.batch_update(batch, value_input_option='USER_ENTERED')
        print(f"   ✅ Batch {j//batch_size + 1} selesai ({len(batch)} cells)")
        if j + batch_size < len(all_cells_to_update):
            import time
            time.sleep(1)
    print(f"✅ Total {len(all_cells_to_update)} cell berhasil diupdate")
else:
    print("ℹ️ Tidak ada cell yang perlu diupdate.")

# === 5. Tambahkan baris prediksi baru ===
existing_records = sheet1.get_all_records()
existing_dates = []
for row in existing_records:
    if "Tanggal" in row and row["Tanggal"]:
        try:
            # Parse format D/M/YYYY (hari/bulan/tahun, tanpa leading zero)
            existing_dates.append(pd.to_datetime(row["Tanggal"], format="%d/%m/%Y"))
        except Exception:
            pass

last_date_in_sheet = max(existing_dates) if existing_dates else None

if last_date_in_sheet:
    print(f"📆 Tanggal terakhir di sheet: {last_date_in_sheet.strftime('%d/%m/%Y')}")

rows_to_add = []
for _, row in df_forecast.iterrows():
    tgl = row["Tanggal"]
    if last_date_in_sheet is None or tgl > last_date_in_sheet:
        # Format: D/M/YYYY (hari/bulan/tahun) TANPA LEADING ZERO
        row_copy = row.copy()
        row_copy["Tanggal"] = f"{tgl.day}/{tgl.month}/{tgl.year}"
        
        cleaned = []
        for val in row_copy.tolist():
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                cleaned.append("")
            else:
                cleaned.append(val)
        rows_to_add.append(cleaned)

if rows_to_add:
    print(f"➕ Menambahkan {len(rows_to_add)} baris baru...")
    sheet1.append_rows(rows_to_add, value_input_option="USER_ENTERED")
    print(f"✅ {len(rows_to_add)} baris prediksi baru ditambahkan.")
else:
    print("ℹ️ Tidak ada baris prediksi baru untuk ditambahkan.")

# === 6. Upload evaluasi model ===
if os.path.exists("model_evaluation.xlsx"):
    df_eval = pd.read_excel("model_evaluation.xlsx", engine="openpyxl")
    df_eval = df_eval.replace([np.inf, -np.inf], np.nan).fillna("")

    # Konversi baris ke list siap upload (serialize datetime → str)
    data_to_upload = []
    for _, row in df_eval.iterrows():
        new_row = []
        for val in row:
            if pd.isna(val):
                new_row.append("")
            elif isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
                new_row.append(val.strftime("%Y-%m-%d"))
            else:
                new_row.append(val)
        data_to_upload.append(new_row)

    try:
        sheet2 = client.open_by_key(SPREADSHEET_ID).worksheet("Sheet2")
    except gspread.WorksheetNotFound:
        client.open_by_key(SPREADSHEET_ID).add_worksheet(title="Sheet2", rows="100", cols="10")
        sheet2 = client.open_by_key(SPREADSHEET_ID).worksheet("Sheet2")

    sheet2.clear()
    sheet2.update(
        [df_eval.columns.values.tolist()] + data_to_upload,
        value_input_option="USER_ENTERED"
    )
    print("📊 Evaluasi model berhasil diupload ke Sheet2 (angka tetap numerik, tanggal diformat)")
else:
    print("ℹ️ File evaluasi model belum ada, skip upload Sheet2")

print("🎉 Upload selesai.")