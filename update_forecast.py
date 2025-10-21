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
    forecast_future,
    evaluate_model_performance,
    get_model_feature_names,
)
from utils.formatting import save_to_excel

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

    # 5. Load model & prediksi
    print("🧠 Memuat model XGBoost dan melakukan prediksi...")
    model = load_model(MODEL_PATH)
    forecast_df, model_input_df = forecast_future(model, df_all, N_FORECAST)

    # 6. Evaluasi performa model (Return)
    print("📊 Mengevaluasi performa model (berdasarkan return)...")
    expected_features = get_model_feature_names(model)
    X_eval = df_all[expected_features].iloc[-30:]
    y_eval = df_all["Return"].iloc[-30:]
    y_pred = model.predict(X_eval)

    metrics = evaluate_model_performance(y_eval, y_pred)
    print(f"✅ RMSE (Return): {metrics['RMSE']:.6f}, MAPE: {metrics['MAPE']:.2%}, R²: {metrics['R2']:.4f}")

    # 7. Evaluasi berbasis harga
    print("\n💰 Mengonversi ke evaluasi berdasarkan harga...")
    eval_df = df_all.iloc[-30:].copy()
    eval_df["Predicted_Return"] = y_pred
    eval_df["Predicted_Price"] = eval_df["Terakhir"].shift(1) * (1 + eval_df["Predicted_Return"])

    valid_df = eval_df.iloc[1:].copy()  # hilangkan baris pertama karena shift

    price_mse = mean_squared_error(valid_df["Terakhir"], valid_df["Predicted_Price"])
    price_rmse = np.sqrt(price_mse)
    price_mae = mean_absolute_error(valid_df["Terakhir"], valid_df["Predicted_Price"])
    price_r2 = r2_score(valid_df["Terakhir"], valid_df["Predicted_Price"])

    print("📈 Evaluasi Model Berdasarkan Harga:")
    print(f"   MSE  : {price_mse:.2f}")
    print(f"   RMSE : {price_rmse:.2f}")
    print(f"   MAE  : {price_mae:.2f}")
    print(f"   R²   : {price_r2:.4f}")

    # 8. Simpan hasil evaluasi gabungan ke Excel
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

    # --- Simpan pakai openpyxl agar format number dan tanggal terdeteksi ---
    wb = Workbook()
    ws = wb.active
    ws.title = "Evaluasi"

    # tulis DataFrame ke sheet
    for r in dataframe_to_rows(df_eval, index=False, header=True):
        ws.append(r)

    # format kolom angka
    for row in ws.iter_rows(min_row=2):  # skip header
        for cell in row:
            if isinstance(cell.value, (int, float)):
                cell.number_format = '[$-421]#,##0.00'

    # format kolom tanggal
    for row in ws.iter_rows(min_row=2, max_col=1):  # kolom pertama (Tanggal_Update)
        for cell in row:
            if isinstance(cell.value, (datetime.date, pd.Timestamp)):
                cell.number_format = 'DD/MM/YYYY'

    wb.save(EVAL_PATH)
    print(f"💾 Hasil evaluasi terbaru disimpan ke {EVAL_PATH}")

    # 9. Simpan hasil forecast utama
    print("📂 Menyimpan hasil prediksi ke Excel...")

    # panggil save_to_excel dengan argumen yang benar: forecast_df (prediksi), model_input_df, df_all_raw
    save_to_excel(forecast_df, model_input_df, df_all_raw, OUTPUT_PATH)

    print("\n🎯 Proses update selesai sepenuhnya.")
