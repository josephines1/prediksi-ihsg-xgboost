import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import holidays
from datetime import timedelta

try:
    import joblib
except ImportError:
    print("Error: paket 'joblib' tidak ditemukan. Install dengan: py -3 -m pip install joblib")
    sys.exit(1)


def load_model(model_path):
    print("🧠 Memuat model...")
    model = joblib.load(model_path)
    return model


def forecast_future(model, df, n_forecast, exclude_holidays=True):
    """
    Buat n_forecast hari bursa (skip weekend + libur Indonesia).
    Menggunakan tanggal terakhir nyata dari df untuk mulai prediksi.
    Mengembalikan (forecast_df, future_df) seperti sebelumnya.
    """
    # Salin & bersihkan
    df = df.copy()
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')
    df['Terakhir'] = pd.to_numeric(df['Terakhir'], errors='coerce')

    # hari ini (normalized)
    today = pd.Timestamp.today().normalize()

    # Ambil tanggal terakhir yang valid: Terakhir tidak NA dan tanggal <= today
    cand = df.loc[df['Terakhir'].notna() & df['Tanggal'].notna() & (df['Tanggal'] <= today), 'Tanggal']
    if cand.empty:
        # fallback: gunakan hari ini (lebih aman daripada memilih tanggal di masa depan)
        print("⚠️ Tidak ditemukan baris historis valid sebelum atau pada hari ini; menggunakan 'today' sebagai titik awal prediksi.")
        last_real_date = today
    else:
        last_real_date = cand.max()

    # Pastikan kita pakai baris terakhir yang valid (tanggal <= last_real_date) sebagai awal untuk membuat lag features dst.
    df_sorted = df.sort_values('Tanggal').reset_index(drop=True)
    df_valid = df_sorted[(df_sorted['Tanggal'].notna()) & (df_sorted['Tanggal'] <= last_real_date) & (df_sorted['Terakhir'].notna())]
    if df_valid.empty:
        raise RuntimeError("Tidak ada baris historis valid untuk memulai prediksi setelah filtering tanggal <= last_real_date.")
    future_df = df_valid.iloc[-5:].copy().reset_index(drop=True)

    # last price untuk kalkulasi iteratif
    if future_df['Terakhir'].isna().all():
        raise RuntimeError("Tidak ada nilai 'Terakhir' valid ditemukan di data untuk memulai forecasting.")
    last_price = future_df['Terakhir'].iloc[-1]

    # holidays Indonesia
    id_holidays = holidays.Indonesia() if exclude_holidays else set()

    # Helper: ambil nama fitur model (asumsi ada fungsi get_model_feature_names() di module lain)
    def get_model_feature_names_local(m):
        # try sklearn wrapper
        try:
            if hasattr(m, 'feature_names_in_'):
                return list(m.feature_names_in_)
        except Exception:
            pass
        try:
            booster = m.get_booster()
            if hasattr(booster, 'feature_names') and booster.feature_names is not None:
                return list(booster.feature_names)
        except Exception:
            pass
        # fallback (sama seperti sebelumnya)
        return ['Terakhir', 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal',
                'Terakhir_lag_1', 'Return_lag_1', 'Terakhir_lag_2', 'Return_lag_2', 'Terakhir_lag_3', 'Return_lag_3',
                'Terakhir_lag_4', 'Return_lag_4', 'Terakhir_lag_5', 'Return_lag_5']

    expected_features = get_model_feature_names_local(model)

    future_predictions = []
    # start from last_real_date (a Timestamp); we'll advance to the next calendar day
    current_date = pd.to_datetime(last_real_date)

    # generate n_forecast business days
    while len(future_predictions) < n_forecast:
        current_date = current_date + timedelta(days=1)

        # skip weekend
        if current_date.weekday() >= 5:
            continue
        # skip holiday
        if exclude_holidays and current_date in id_holidays:
            continue

        # pastikan fitur ada di future_df
        missing = [f for f in expected_features if f not in future_df.columns]
        if missing:
            raise RuntimeError(f"Input data is missing features required by the model: {missing}")

        X_last = future_df[expected_features].iloc[[-1]]
        next_return = float(model.predict(X_last)[0])
        next_price = last_price * (1 + next_return)

        future_predictions.append({
            'Tanggal': current_date,
            'Predicted_Return': next_return,
            'Predicted_Price': next_price
        })

        # tambahkan baris baru ke future_df untuk prediksi iteratif berikutnya
        new_row = future_df.iloc[-1].copy()
        new_row['Tanggal'] = current_date
        new_row['Terakhir'] = next_price
        new_row['Return'] = next_return
        future_df.loc[len(future_df)] = new_row

        last_price = next_price

    forecast_df = pd.DataFrame(future_predictions)
    future_df['is_forecast'] = False
    if len(forecast_df) > 0:
        future_df.loc[future_df.index[-len(forecast_df):], 'is_forecast'] = True

    return forecast_df, future_df

def evaluate_model_performance(y_true, y_pred):
    """Evaluasi performa model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }

def get_model_feature_names(model):
    try:
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
    except Exception:
        pass
    try:
        booster = model.get_booster()
        if hasattr(booster, 'feature_names') and booster.feature_names is not None:
            return list(booster.feature_names)
    except Exception:
        pass
    # fallback — daftar fitur training
    return ['Terakhir', 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30',
            'RSI', 'MACD', 'MACD_signal',
            'Terakhir_lag_1', 'Return_lag_1',
            'Terakhir_lag_2', 'Return_lag_2',
            'Terakhir_lag_3', 'Return_lag_3',
            'Terakhir_lag_4', 'Return_lag_4',
            'Terakhir_lag_5', 'Return_lag_5']

