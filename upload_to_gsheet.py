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

# === 2. Konfigurasi Spreadsheet ===
SPREADSHEET_ID = "13EfYovEBQHg19pDY21I8RV7RR4Mo1SNSRzNckg_dwhw"

# === 3. Load forecast terbaru ===
try:
    df_forecast = pd.read_excel("ihsg_forecast.xlsx")
    print(f"📊 File ihsg_forecast.xlsx berisi {len(df_forecast)} baris")
except FileNotFoundError:
    raise SystemExit("❌ File 'ihsg_forecast.xlsx' tidak ditemukan!")

# Bersihkan dan format data
numeric_cols = ["Terakhir", "Pembukaan", "Tertinggi", "Terendah"]
for col in numeric_cols:
    if col in df_forecast.columns:
        df_forecast[col] = pd.to_numeric(df_forecast[col], errors="coerce").round(2)

df_forecast = df_forecast.replace([np.inf, -np.inf], np.nan)
df_forecast = df_forecast.where(pd.notnull(df_forecast), None)

# Parse tanggal jadi datetime object dulu
df_forecast["Tanggal"] = pd.to_datetime(df_forecast["Tanggal"])

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

print(f"📋 Sheet1 memiliki {len(records)} baris data")

# === 5. Update KEMARIN jadi Historis ===
yesterday = today - datetime.timedelta(days=1)

# Kalau kemarin weekend/libur, mundur sampai hari bursa terakhir
target_date = yesterday
for _ in range(7):
    if is_trading_day(target_date):
        break
    print(f"   {target_date.strftime('%d/%m/%Y')} bukan hari bursa, mundur...")
    target_date -= datetime.timedelta(days=1)

# Format: D/M/YYYY (hari/bulan/tahun) TANPA LEADING ZERO
target_date_str = f"{target_date.day}/{target_date.month}/{target_date.year}"
print(f"🎯 Target update: {target_date.strftime('%d %B %Y')} → {target_date_str}")

# Cari baris di sheet
found_row_index = None
for i, row in enumerate(records, start=2):
    if len(row) > tanggal_idx:
        row_date = row[tanggal_idx].strip()
        if row_date == target_date_str:
            found_row_index = i
            print(f"   ✓ Ketemu di baris {i}")
            break

if found_row_index:
    current_status = records[found_row_index - 2][status_idx] if status_idx is not None and len(records[found_row_index - 2]) > status_idx else ""
    
    if current_status != "Historis":
        print(f"✏️ Update baris {found_row_index} → Status = Historis")
        
        if status_idx is not None:
            sheet1.update_cell(found_row_index, status_idx + 1, "Historis")
            print(f"✅ Berhasil update {target_date_str} jadi Historis")

        # === Lengkapi data kosong untuk baris yang baru diubah jadi Historis ===
        forecast_row = df_forecast.loc[df_forecast["Tanggal"].dt.date == target_date]
        if not forecast_row.empty:
            forecast_row = forecast_row.iloc[0]

            # Mapping kolom: nama di Sheet → nama di DataFrame
            col_mapping = {
                "Terakhir": "Terakhir",
                "Pembukaan": "Pembukaan",
                "Tertinggi": "Tertinggi",
                "Terendah": "Terendah",
                "Vol.": "Vol.",
                "Perubahan%": "Perubahan%"
            }

            for col_name, df_col in col_mapping.items():
                if col_name in headers:
                    col_idx = headers.index(col_name) + 1  # +1 karena index 0-based
                    new_val = forecast_row.get(df_col, None)

                    # Kolom 'Terakhir' → langsung timpa tanpa cek kosong
                    if col_name == "Terakhir":
                        if pd.notnull(new_val):
                            sheet1.update_cell(found_row_index, col_idx, new_val)
                            print(f"   ✳️ Kolom 'Terakhir' diperbarui dengan nilai dari forecast ({new_val})")
                        else:
                            print(f"   ⚠️ Nilai 'Terakhir' di forecast kosong, tidak diperbarui")

                    # Kolom lain → isi hanya jika kosong
                    else:
                        current_val = (
                            records[found_row_index - 2][col_idx - 1]
                            if len(records[found_row_index - 2]) >= col_idx
                            else ""
                        )
                        if (current_val == "" or current_val is None) and pd.notnull(new_val):
                            sheet1.update_cell(found_row_index, col_idx, new_val)
                            print(f"   🔄 Kolom '{col_name}' dilengkapi dengan nilai dari forecast ({new_val})")
                        elif current_val != "" and current_val is not None:
                            print(f"   ℹ️ Kolom '{col_name}' sudah terisi ({current_val}), skip.")
                        else:
                            print(f"   ⚠️ Kolom '{col_name}' tetap kosong (tidak ada nilai di forecast)")
        else:
            print(f"⚠️ Data {target_date_str} tidak ditemukan di ihsg_forecast.xlsx — tidak bisa melengkapi nilai.")
    else:
        print(f"ℹ️ Baris {found_row_index} sudah Historis, skip.")
else:
    print(f"⚠️ Tanggal {target_date_str} tidak ditemukan di sheet")

# === 6. Tambahkan baris prediksi baru ===
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