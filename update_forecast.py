"""
update_forecast.py
--------------------------------
Script otomatis untuk:
1. Ambil data terbaru IHSG dari Yahoo Finance.
2. Gabungkan dengan dataset historis lokal (Investing.com CSV).
3. Load model XGBoost terlatih dari Hugging Face (atau file lokal).
4. Prediksi n hari ke depan.
5. Simpan hasil ke Excel (mis. OneDrive/Google Drive).
"""

import pandas as pd
import numpy as np
import sys

# Guard imports to provide clearer instructions if a dependency is missing
try:
    import yfinance as yf
except ImportError:
    print("Error: paket 'yfinance' tidak ditemukan. Install dependensi dengan: py -3 -m pip install yfinance pandas numpy joblib xgboost openpyxl")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("Error: paket 'joblib' tidak ditemukan. Install dependensi dengan: py -3 -m pip install joblib")
    sys.exit(1)
from datetime import datetime, timedelta
import os

# ===============================
# KONFIGURASI
# ===============================
LOCAL_DATASET1_PATH = "data/IHSG 1993-2013.csv"
LOCAL_DATASET2_PATH = "data/IHSG 2013-2025⁄9⁄26.csv"
MODEL_PATH = "model/xgboost_ihsg_model.pkl"
OUTPUT_PATH = "./ihsg_forecast.xlsx"
TICKER = "^JKSE"
N_FORECAST = 5

COLS_TO_DROP = ['Tanggal']

# ===============================
# UTILITAS TAMBAHAN
# ===============================
def parse_euro_number(x):
    """
    Parse number in formats:
    '8.125,20' -> 8125.20
    '8125,20'  -> 8125.20
    '8125.20'  -> 8125.20
    numeric -> float as is
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    s = str(x).strip()
    if s == '':
        return np.nan
    if '.' in s and ',' in s:
        s2 = s.replace('.', '').replace(',', '.')
        try:
            return float(s2)
        except:
            return np.nan
    if ',' in s and '.' not in s:
        s2 = s.replace(',', '.')
        try:
            return float(s2)
        except:
            return np.nan
    try:
        return float(s)
    except:
        s3 = ''.join(ch for ch in s if ch.isdigit() or ch in '.,-')
        s3 = s3.replace(',', '.')
        try:
            return float(s3)
        except:
            return np.nan

def fmt_price_eu(v):
    if pd.isna(v):
        return ""
    v = round(v, 2)  # dua desimal
    s = f"{v:,.2f}"  # default: 1,234.56
    # swap ke format CSV lokal: '.' ribuan, ',' desimal
    s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
    return s

def fmt_volume_eu(v):
    if pd.isna(v):
        return ""
    try:
        val = float(v)
    except:
        return ""
    if abs(val) >= 1e9:
        out = val / 1e9
        s = f"{out:.2f}".replace('.', ',') + 'B'
    elif abs(val) >= 1e6:
        out = val / 1e6
        s = f"{out:.2f}".replace('.', ',') + 'M'
    else:
        return fmt_price_eu(val)
    return s

# ===============================
# UTILITAS
# ===============================
def parse_volume(vol_str):
    """Ubah string seperti '4,11M' atau '35,20B' jadi angka"""
    if pd.isna(vol_str):
        return np.nan
    s = str(vol_str).replace('.', '').replace(',', '.').strip()
    multiplier = 1
    if s.endswith('M'):
        multiplier = 1e6
        s = s[:-1]
    elif s.endswith('B'):
        multiplier = 1e9
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan


def normalize_local(df_local):
    """Normalisasi dataset Investing.com"""
    df_local = df_local.copy()
    df_local.columns = [c.strip() for c in df_local.columns]

    for col in ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']:
        df_local[col] = df_local[col].apply(parse_euro_number)

    df_local['Vol.'] = df_local['Vol.'].apply(parse_volume)
    df_local['Tanggal'] = pd.to_datetime(df_local['Tanggal'], format='%d/%m/%Y')

    df_local['Perubahan%'] = (
        df_local['Perubahan%']
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.replace('+', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    df_local['Perubahan%'] = pd.to_numeric(df_local['Perubahan%'], errors='coerce')


    return df_local


def fetch_latest_data():
    """Ambil data terbaru dari Yahoo Finance"""
    print("📈 Mengambil data IHSG terbaru dari Yahoo Finance...")
    today = datetime.today()
    start_date = "1990-01-01"
    df = yf.download(TICKER, start=start_date, end=today)
    df = df.reset_index().rename(columns={
        'Date': 'Tanggal',
        'Open': 'Pembukaan',
        'High': 'Tertinggi',
        'Low': 'Terendah',
        'Close': 'Terakhir',
        'Volume': 'Vol.'
    })
    df = df[['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.']]
    df['Perubahan%'] = df['Terakhir'].pct_change() * 100
    df['Vol.'] = df['Vol.'].replace(0, np.nan)
    return df


def merge_yf_and_local(df_yf, df1_local, df2_local):
    """Gabungkan YF + dua CSV lokal"""
    print("🔄 Menggabungkan dataset lokal & Yahoo Finance...")

    df_local = pd.concat([df1_local, df2_local], ignore_index=True)
    df_local = normalize_local(df_local)
    df_yf = df_yf.copy()
    # Jika yfinance mengembalikan MultiIndex columns (kadang terjadi), flatten dulu
    if isinstance(df_yf.columns, pd.MultiIndex):
        # gabungkan level kolom dengan underscore, abaikan level kosong
        df_yf.columns = ["_".join([str(part) for part in col if part not in ("", None)]) for col in df_yf.columns]
        # Coba map nama kolom umum ke nama bahasa Indonesia yang kita pakai
        rename_map = {}
        for c in df_yf.columns:
            low = c.lower()
            if low.endswith('date') or low.endswith('tanggal'):
                rename_map[c] = 'Tanggal'
            elif 'close' in low or 'adj close' in low or 'last' in low:
                rename_map[c] = 'Terakhir'
            elif 'open' in low:
                rename_map[c] = 'Pembukaan'
            elif 'high' in low:
                rename_map[c] = 'Tertinggi'
            elif 'low' in low:
                rename_map[c] = 'Terendah'
            elif 'vol' in low:
                rename_map[c] = 'Vol.'
        if rename_map:
            df_yf.rename(columns=rename_map, inplace=True)

    # Pastikan kolom tanggal ada dan bertipe datetime
    if 'Date' in df_yf.columns and 'Tanggal' not in df_yf.columns:
        df_yf.rename(columns={'Date': 'Tanggal'}, inplace=True)
    if 'Tanggal' in df_yf.columns:
        df_yf['Tanggal'] = pd.to_datetime(df_yf['Tanggal'])

    # Jika beberapa kolom penting tidak ada (mis. karena nama bahasa Inggris), coba cari kandidat kolom
    def map_from_candidates(target, candidates):
        if target in df_yf.columns:
            return True
        for c in df_yf.columns:
            low = str(c).lower()
            for pat in candidates:
                if pat in low:
                    df_yf[target] = df_yf[c]
                    print(f"ℹ️ Kolom '{target}' dibuat dari '{c}' (cocok '{pat}')")
                    return True
        return False

    # Mapping dasar: close/open/high/low/volume/date
    map_from_candidates('Terakhir', ['close', 'adj close', 'last'])
    map_from_candidates('Pembukaan', ['open'])
    map_from_candidates('Tertinggi', ['high'])
    map_from_candidates('Terendah', ['low'])
    map_from_candidates('Vol.', ['volume', 'vol'])


    merged = pd.merge(df_yf, df_local[['Tanggal', 'Vol.']], on='Tanggal', how='left', suffixes=('', '_local'))
    merged['Vol.'] = np.where(merged['Vol.'].isna(), merged['Vol._local'], merged['Vol.'])
    merged.drop(columns=['Vol._local'], inplace=True)

    merged = merged.sort_values('Tanggal').drop_duplicates(subset='Tanggal', keep='last')
    merged.reset_index(drop=True, inplace=True)
    # Pastikan kolom-kolom penting ada; lakukan fallback mapping jika perlu
    candidates_map = {
        'Terakhir': ['terakhir', 'close', 'adj close', 'adjclose', 'close_adj', 'last', 'price'],
        'Pembukaan': ['pembukaan', 'open'],
        'Tertinggi': ['tertinggi', 'high', 'max'],
        'Terendah': ['terendah', 'low', 'min'],
        'Vol.': ['vol.', 'volume', 'vol']
    }

    for target, candidates in candidates_map.items():
        if target in merged.columns:
            continue
        found = False
        for c in merged.columns:
            low = str(c).lower()
            for pat in candidates:
                if pat in low:
                    merged[target] = merged[c]
                    print(f"ℹ️ Setelah merge: kolom '{target}' dibuat dari '{c}' (cocok '{pat}')")
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"⚠️ Kolom '{target}' tidak ditemukan setelah merge dan tidak dapat dibuat otomatis.")

    # Coerce Terakhir dan Vol. ke numeric
    if 'Terakhir' in merged.columns:
        # Cek tipe data, jika string kemungkinan data lokal, parse. 
        if merged['Terakhir'].dtype == object:
            merged['Terakhir'] = merged['Terakhir'].apply(parse_euro_number)
        else:
            merged['Terakhir'] = merged['Terakhir'].astype(float)

    if 'Vol.' in merged.columns:
        merged['Vol.'] = pd.to_numeric(merged['Vol.'], errors='coerce')

    # Setelah fallback mapping selesai, hapus kolom-kolom duplikat yang mengandung ticker suffix seperti '_^JKSE'
    cols_to_remove = [c for c in merged.columns if '^jkse' in str(c).lower()]
    if cols_to_remove:
        # Jangan hapus jika nama kolom tepat sama dengan canonical (hanya hapus suffixed)
        to_drop = [c for c in cols_to_remove if c not in ['Terakhir','Pembukaan','Tertinggi','Terendah','Vol.','Perubahan%','Tanggal']]
        if to_drop:
            print(f"ℹ️ Menghapus kolom duplikat setelah mapping: {to_drop}")
            merged.drop(columns=to_drop, inplace=True)

    print(f"✅ Data tergabung: {len(merged)} baris total.")
    return merged


# ===============================
# FEATURE ENGINEERING
# ===============================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_features(df):
    df['Return'] = df['Terakhir'].pct_change()
    df['EMA_9'] = df['Terakhir'].ewm(span=9).mean()
    df['SMA_5'] = df['Terakhir'].rolling(5).mean()
    df['SMA_10'] = df['Terakhir'].rolling(10).mean()
    df['SMA_15'] = df['Terakhir'].rolling(15).mean()
    df['SMA_30'] = df['Terakhir'].rolling(30).mean()
    df['RSI'] = compute_rsi(df['Terakhir'])
    # MACD (12,26) and MACD signal (9)
    ema_12 = df['Terakhir'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Terakhir'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Lag features for Terakhir and Return (1..5)
    for lag in range(1, 6):
        df[f'Terakhir_lag_{lag}'] = df['Terakhir'].shift(lag)
        df[f'Return_lag_{lag}'] = df['Return'].shift(lag)
    df = df.dropna().reset_index(drop=True)
    # Reorder columns to match model's expected feature order if possible
    expected_order = ['Terakhir', 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal'] + \
        [f'Terakhir_lag_{i}' for i in range(1,6)] + [f'Return_lag_{i}' for i in range(1,6)]
    existing = [c for c in expected_order if c in df.columns]
    other = [c for c in df.columns if c not in existing]
    df = df[existing + other]
    return df


# ===============================
# FORECASTING
# ===============================
def load_model(model_path):
    print("🧠 Memuat model...")
    model = joblib.load(model_path)
    return model


def forecast_future(model, df, n_forecast):
    # future_df is the working dataset fed into the model; we'll return it (with predicted rows)
    future_df = df.iloc[-5:].copy().reset_index(drop=True)
    last_price = future_df['Terakhir'].iloc[-1]
    future_predictions = []
    future_dates = []

    current_date = future_df['Tanggal'].iloc[-1]

    def get_model_feature_names(m):
        # Try several ways to get expected feature names from an XGBoost model
        try:
            # sklearn wrapper
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
        # Fallback: common expected features used during training
        return ['Terakhir', 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal',
                'Terakhir_lag_1', 'Return_lag_1', 'Terakhir_lag_2', 'Return_lag_2', 'Terakhir_lag_3', 'Return_lag_3',
                'Terakhir_lag_4', 'Return_lag_4', 'Terakhir_lag_5', 'Return_lag_5']

    while len(future_predictions) < n_forecast:
        current_date += timedelta(days=1)
        if current_date.weekday() >= 5:
            continue

        # Build X_last using exactly the model's expected feature names (and order)
        expected_features = get_model_feature_names(model)
        missing = [f for f in expected_features if f not in future_df.columns]
        if missing:
            raise RuntimeError(f"Input data is missing features required by the model: {missing}")
        X_last = future_df[expected_features].iloc[[-1]]
        next_return = model.predict(X_last)[0]
        next_price = last_price * (1 + next_return)

        future_predictions.append({
            'Tanggal': current_date,
            'Predicted_Return': next_return,
            'Predicted_Price': next_price
        })

        new_row = future_df.iloc[-1].copy()
        new_row['Tanggal'] = current_date
        new_row['Terakhir'] = next_price
        new_row['Return'] = next_return
        future_df.loc[len(future_df)] = new_row

        last_price = next_price

    forecast_df = pd.DataFrame(future_predictions)
    # Mark forecast rows inside the model input df for clarity
    future_df['is_forecast'] = False
    # the last len(future_predictions) rows are forecasted
    if len(future_predictions) > 0:
        future_df.loc[future_df.index[-len(future_predictions):], 'is_forecast'] = True

    print("✅ Prediksi selesai.")
    return forecast_df, future_df


def save_to_excel(forecast_df, model_input_df, df_all_raw, output_path):
    """
    Save a single-sheet Excel:
    - Historical rows: df_all_raw
    - Forecast rows: forecast_df
    """
    df_hist = df_all_raw.copy()
    if 'Tanggal' in df_hist.columns:
        df_hist['Tanggal'] = pd.to_datetime(df_hist['Tanggal'], errors='coerce')
    out_cols = ['Tanggal','Terakhir','Pembukaan','Tertinggi','Terendah','Vol.','Perubahan%']
    for c in out_cols:
        if c not in df_hist.columns:
            df_hist[c] = np.nan

    hist_rows = []
    for _, r in df_hist.sort_values('Tanggal').iterrows():
        date = r['Tanggal']
        date_str = pd.to_datetime(date).strftime('%d/%m/%Y') if pd.notna(date) else ''
        row = {
            'Tanggal': date_str,
            'Terakhir': fmt_price_eu(r['Terakhir']),
            'Pembukaan': fmt_price_eu(r['Pembukaan']),
            'Tertinggi': fmt_price_eu(r['Tertinggi']),
            'Terendah': fmt_price_eu(r['Terendah']),
            'Vol.': fmt_volume_eu(r['Vol.']),
            'Perubahan%': (f"{r['Perubahan%']:.2f}".replace('.', ',') + '%') if pd.notna(r['Perubahan%']) else ''
        }
        hist_rows.append(row)

    fore_rows = []
    for _, fr in forecast_df.iterrows():
        date = fr['Tanggal']
        date_str = pd.to_datetime(date).strftime('%d/%m/%Y') if pd.notna(date) else ''
        fore_rows.append({
            'Tanggal': date_str,
            'Terakhir': fmt_price_eu(fr['Predicted_Price']),
            'Pembukaan': '',
            'Tertinggi': '',
            'Terendah': '',
            'Vol.': '',
            'Perubahan%': ''
        })

    out_df = pd.DataFrame(hist_rows + fore_rows, columns=out_cols)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
            out_df.to_excel(writer, index=False, sheet_name='Sheet1')
        print(f"💾 File tersimpan: {output_path}")
    except PermissionError:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fallback = os.path.join(os.path.dirname(__file__), f"ihsg_forecast_fallback_{ts}.xlsx")
        with pd.ExcelWriter(fallback, engine='openpyxl', mode='w') as writer:
            out_df.to_excel(writer, index=False, sheet_name='Sheet1')
        print(f"⚠️ File terkunci, disimpan ke fallback: {fallback}")


# ===============================
# MAIN FLOW
# ===============================
if __name__ == "__main__":
    print("🚀 Menjalankan update_forecast.py")

    # 1. Ambil data terbaru
    df_yf = fetch_latest_data()

    # 2. Baca dataset lokal
    df1 = pd.read_csv(LOCAL_DATASET1_PATH)
    df2 = pd.read_csv(LOCAL_DATASET2_PATH)

    # 3. Merge semuanya
    df_all_raw = merge_yf_and_local(df_yf, df1, df2)

    # 4. Feature engineering
    df_all = add_features(df_all_raw)

    # 5. Load model & prediksi
    model = load_model(MODEL_PATH)
    forecast_df, model_input_df = forecast_future(model, df_all, N_FORECAST)

    # 6. Simpan seluruh historis + prediksi
    save_to_excel(forecast_df, model_input_df, df_all_raw, OUTPUT_PATH)

    print("🎯 Proses selesai sepenuhnya.")
