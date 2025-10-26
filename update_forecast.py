import sys
import os
import datetime
import holidays
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from openpyxl import load_workbook, Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from utils.data_utils import fetch_latest_data, merge_yf_and_local
from utils.features import add_features
from utils.forecast_utils import (
    load_model,
    forecast_future_incremental,
    evaluate_model_performance,
    get_model_feature_names,
)

# ===============================
# CEK LIBUR NASIONAL & AKHIR PEKAN
# ===============================
today = datetime.date.today()
id_holidays = holidays.Indonesia()

if today.weekday() >= 5 or today in id_holidays:
    reason = "akhir pekan" if today.weekday() >= 5 else "libur nasional"
    print(f"⏸️ Hari ini ({today}) adalah {reason}. Prediksi otomatis dilewati.")
    sys.exit(0)

# ===============================
# KONFIGURASI
# ===============================
LOCAL_DATASET1_PATH = "data/IHSG 1993-2013.csv"
LOCAL_DATASET2_PATH = "data/IHSG 2013-2025⁄9⁄26.csv"
MODEL_PATH = "model/xgboost_ihsg_model.pkl"
OUTPUT_PATH = "./ihsg_forecast.xlsx"
EVAL_PATH = "./model_evaluation.xlsx"
N_FORECAST = 30

# ===============================
# MAIN FLOW
# ===============================
if __name__ == "__main__":
    print(f"🚀 Menjalankan update_forecast.py pada {today}")

    # 1. Ambil data terbaru
    print("📥 Mengambil data IHSG terbaru dari Yahoo Finance...")
    df_yf = fetch_latest_data()

    # 2. Baca dataset historis lokal
    print("📂 Membaca dataset historis lokal...")
    df1 = pd.read_csv(LOCAL_DATASET1_PATH)
    df2 = pd.read_csv(LOCAL_DATASET2_PATH)

    # 3. Gabungkan semuanya
    print("🧩 Menggabungkan semua dataset...")
    df_all_raw = merge_yf_and_local(df_yf, df1, df2)

    # 4. Feature engineering
    print("⚙️ Menambahkan fitur teknikal...")
    df_all = add_features(df_all_raw)

    # 5. Load model
    print("🧠 Memuat model XGBoost...")
    model = load_model(MODEL_PATH)

    # 6. Baca file Excel yang sudah ada (kalau ada)
    existing_df = None
    if os.path.exists(OUTPUT_PATH):
        print(f"📖 Membaca file existing: {OUTPUT_PATH}")
        try:
            existing_df = pd.read_excel(OUTPUT_PATH, sheet_name="Sheet1")
            existing_df['Tanggal'] = pd.to_datetime(existing_df['Tanggal'], dayfirst=True, errors='coerce')
            print(f"   ✓ File existing berisi {len(existing_df)} baris")
        except Exception as e:
            print(f"   ⚠️ Error membaca file existing: {e}")
            existing_df = None

    # 7. Generate prediksi incremental
    print("🔮 Membuat prediksi incremental...")
    updated_df = forecast_future_incremental(model, df_all, df_all_raw, existing_df, N_FORECAST)

    # 8. Evaluasi performa model (Return) - hanya untuk 30 hari terakhir
    print("📊 Mengevaluasi performa model (berdasarkan return)...")
    expected_features = get_model_feature_names(model)
    X_eval = df_all[expected_features].iloc[-30:]
    y_eval = df_all["Return"].iloc[-30:]
    y_pred = model.predict(X_eval)

    metrics = evaluate_model_performance(y_eval, y_pred)
    print(f"✅ RMSE (Return): {metrics['RMSE']:.6f}, MAPE: {metrics['MAPE']:.2%}, R²: {metrics['R2']:.4f}")

    # 9. Evaluasi berbasis harga
    print("\n💰 Mengonversi ke evaluasi berdasarkan harga...")
    eval_df = df_all.iloc[-30:].copy()
    eval_df["Predicted_Return"] = y_pred
    eval_df["Predicted_Price"] = eval_df["Terakhir"].shift(1) * (1 + eval_df["Predicted_Return"])

    valid_df = eval_df.iloc[1:].copy()

    price_mse = mean_squared_error(valid_df["Terakhir"], valid_df["Predicted_Price"])
    price_rmse = np.sqrt(price_mse)
    price_mae = mean_absolute_error(valid_df["Terakhir"], valid_df["Predicted_Price"])
    price_r2 = r2_score(valid_df["Terakhir"], valid_df["Predicted_Price"])

    print("📈 Evaluasi Model Berdasarkan Harga:")
    print(f"   MSE  : {price_mse:.2f}")
    print(f"   RMSE : {price_rmse:.2f}")
    print(f"   MAE  : {price_mae:.2f}")
    print(f"   R²   : {price_r2:.4f}")

    # 10. Simpan hasil evaluasi
    today_date = pd.Timestamp.today().date()

    eval_data = {
        "Tanggal_Update": [today_date],
        "RMSE": [metrics["RMSE"]],
        "MAPE": [metrics["MAPE"]],
        "R2": [metrics["R2"]],
        "MSE": [metrics["MSE"]],
        "Jumlah_Data_Evaluasi": [len(valid_df)],
        "MSE_Price": [price_mse],
        "RMSE_Price": [price_rmse],
        "MAE_Price": [price_mae],
        "R2_Price": [price_r2],
    }

    df_eval = pd.DataFrame(eval_data)

    wb = Workbook()
    ws = wb.active
    ws.title = "Evaluasi"

    for r in dataframe_to_rows(df_eval, index=False, header=True):
        ws.append(r)

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, (int, float)):
                cell.number_format = '[$-421]#,##0.00'

    for row in ws.iter_rows(min_row=2, max_col=1):
        for cell in row:
            if isinstance(cell.value, (datetime.date, pd.Timestamp)):
                cell.number_format = 'DD/MM/YYYY'

    wb.save(EVAL_PATH)
    print(f"💾 Hasil evaluasi terbaru disimpan ke {EVAL_PATH}")

    # 11. Simpan hasil forecast
    print("📂 Menyimpan hasil prediksi ke Excel...")
    
    wb_output = Workbook()
    ws_output = wb_output.active
    ws_output.title = "Sheet1"

    for r in dataframe_to_rows(updated_df, index=False, header=True):
        ws_output.append(r)

    # Format kolom
    for row in ws_output.iter_rows(min_row=2):
        # Tanggal
        row[0].number_format = 'DD/MM/YYYY'
        
        # Kolom angka (B-E dan H)
        for i in [1, 2, 3, 4, 7]:
            cell = row[i]
            try:
                if cell.value not in (None, '', ' '):
                    cell.value = float(cell.value)
                    cell.number_format = '[$-421]#,##0.00'
            except Exception:
                pass

    wb_output.save(OUTPUT_PATH)
    print(f"💾 File tersimpan: {OUTPUT_PATH}")

    print("\n🎯 Proses update selesai sepenuhnya.")