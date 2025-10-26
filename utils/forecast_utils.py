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


def forecast_future_incremental(model, df_all, df_all_raw, existing_df, n_forecast, exclude_holidays=True):
    """
    Update incremental: hanya generate prediksi untuk data baru.
    
    Logic:
    1. Kalau file existing ada, ambil data yang sudah ada
    2. Cek tanggal mana yang belum punya prediksi (kolom Terakhir (Prediksi) kosong)
    3. Generate prediksi hanya untuk tanggal yang belum ada prediksinya
    4. Tambah prediksi 30 hari ke depan
    
    Returns:
        DataFrame final dengan struktur: Tanggal | Terakhir | ... | Terakhir (Prediksi)
    """
    
    def fmt_volume(x):
        if pd.isna(x):
            return ""
        try:
            v = float(x)
        except Exception:
            return ""
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f}B"
        elif abs(v) >= 1e6:
            return f"{v/1e6:.2f}M"
        else:
            return f"{v:.2f}"
    
    # Siapkan data historis
    df_hist = df_all_raw.copy()
    df_hist['Tanggal'] = pd.to_datetime(df_hist.get('Tanggal'), dayfirst=True, errors='coerce')
    df_hist = df_hist.dropna(subset=['Tanggal']).sort_values('Tanggal').reset_index(drop=True)
    
    # Helper: get feature names
    def get_model_feature_names_local(m):
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
        return ['Terakhir', 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal',
                'Terakhir_lag_1', 'Return_lag_1', 'Terakhir_lag_2', 'Return_lag_2', 'Terakhir_lag_3', 'Return_lag_3',
                'Terakhir_lag_4', 'Return_lag_4', 'Terakhir_lag_5', 'Return_lag_5']
    
    expected_features = get_model_feature_names_local(model)
    
    # === Kalau file existing ada, pakai data yang sudah ada ===
    if existing_df is not None and len(existing_df) > 0:
        print("   📋 Menggunakan data existing sebagai base...")
        result_df = existing_df.copy()
        
        # Cari baris yang belum ada prediksi (Terakhir (Prediksi) kosong)
        mask_empty = result_df['Terakhir (Prediksi)'].isna() | (result_df['Terakhir (Prediksi)'] == '') | (result_df['Terakhir (Prediksi)'] == 0)
        rows_need_prediction = result_df[mask_empty].copy()
        
        print(f"   🎯 {len(rows_need_prediction)} baris perlu diupdate prediksinya")
        
        # Generate prediksi untuk baris yang kosong
        if len(rows_need_prediction) > 0:
            for idx, row in rows_need_prediction.iterrows():
                tgl = row['Tanggal']
                
                # Cari baris di df_all yang sesuai
                df_all_copy = df_all.copy()
                df_all_copy['Tanggal'] = pd.to_datetime(df_all_copy['Tanggal'], dayfirst=True, errors='coerce')
                matching = df_all_copy[df_all_copy['Tanggal'] == tgl]
                
                if len(matching) > 0:
                    try:
                        X = matching[expected_features]
                        pred_return = float(model.predict(X)[0])
                        pred_price = matching['Terakhir'].iloc[0] * (1 + pred_return)
                        
                        # Update di result_df
                        result_df.loc[idx, 'Terakhir (Prediksi)'] = pred_price
                        print(f"      ✓ Updated {tgl.strftime('%d/%m/%Y')}: {pred_price:.2f}")
                    except Exception as e:
                        # Untuk baris awal yang belum punya lag features, gunakan nilai Terakhir sebagai prediksi
                        try:
                            actual_price = matching['Terakhir'].iloc[0]
                            if pd.notna(actual_price):
                                result_df.loc[idx, 'Terakhir (Prediksi)'] = actual_price
                                print(f"      ⚠️ {tgl.strftime('%d/%m/%Y')}: Pakai nilai aktual (lag incomplete)")
                        except:
                            print(f"      ⚠️ Skip {tgl.strftime('%d/%m/%Y')}: {str(e)[:50]}")
                        continue
        
        # Cek tanggal terakhir di data historis (yang punya nilai Terakhir)
        hist_rows = result_df[result_df['Terakhir'].notna() & (result_df['Terakhir'] != '')]
        if len(hist_rows) > 0:
            last_hist_date = hist_rows['Tanggal'].max()
        else:
            last_hist_date = result_df['Tanggal'].min()
        
        print(f"   📅 Tanggal terakhir historis: {last_hist_date.strftime('%d/%m/%Y')}")
        
        # Hapus semua baris prediksi masa depan yang lama (yang Terakhir kosong)
        result_df = result_df[result_df['Tanggal'] <= last_hist_date].copy()
        print(f"   🗑️ Hapus prediksi lama, tersisa {len(result_df)} baris")
        
    else:
        # Kalau file existing tidak ada, buat dari awal
        print("   ⚠️ File existing tidak ada, buat dari awal...")
        result_df = pd.DataFrame(columns=['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%', 'Terakhir (Prediksi)'])
        
        # Isi dengan data historis
        for _, r in df_hist.iterrows():
            result_df = pd.concat([result_df, pd.DataFrame([{
                'Tanggal': r['Tanggal'],
                'Terakhir': pd.to_numeric(r.get('Terakhir', np.nan), errors='coerce'),
                'Pembukaan': pd.to_numeric(r.get('Pembukaan', np.nan), errors='coerce'),
                'Tertinggi': pd.to_numeric(r.get('Tertinggi', np.nan), errors='coerce'),
                'Terendah': pd.to_numeric(r.get('Terendah', np.nan), errors='coerce'),
                'Vol.': fmt_volume(r.get('Vol.', np.nan)),
                'Perubahan%': (f"{r['Perubahan%']:.2f}%" if pd.notna(r.get('Perubahan%')) else ''),
                'Terakhir (Prediksi)': ''
            }])], ignore_index=True)
        
        last_existing_date = result_df['Tanggal'].max()
    
    # === Generate prediksi 30 hari ke depan ===
    print(f"   🔮 Membuat prediksi {n_forecast} hari ke depan...")
    
    # Ambil data terakhir untuk mulai prediksi
    df_sorted = df_all.copy()
    df_sorted['Tanggal'] = pd.to_datetime(df_sorted['Tanggal'], dayfirst=True, errors='coerce')
    df_sorted = df_sorted.sort_values('Tanggal').reset_index(drop=True)
    
    today = pd.Timestamp.today().normalize()
    df_valid = df_sorted[(df_sorted['Tanggal'].notna()) & (df_sorted['Tanggal'] <= today) & (df_sorted['Terakhir'].notna())]
    
    if df_valid.empty:
        print("   ⚠️ Tidak ada data valid untuk prediksi masa depan")
        return result_df
    
    future_df = df_valid.iloc[-5:].copy().reset_index(drop=True)
    last_price = future_df['Terakhir'].iloc[-1]
    last_hist_date = df_valid['Tanggal'].max()
    
    id_holidays = holidays.Indonesia() if exclude_holidays else set()
    current_date = last_hist_date
    
    future_predictions = []
    while len(future_predictions) < n_forecast:
        current_date = current_date + timedelta(days=1)
        
        if current_date.weekday() >= 5:
            continue
        if exclude_holidays and current_date in id_holidays:
            continue
        
        X_last = future_df[expected_features].iloc[[-1]]
        next_return = float(model.predict(X_last)[0])
        next_price = last_price * (1 + next_return)
        
        future_predictions.append({
            'Tanggal': current_date,
            'Terakhir': '',
            'Pembukaan': '',
            'Tertinggi': '',
            'Terendah': '',
            'Vol.': '',
            'Perubahan%': '',
            'Terakhir (Prediksi)': next_price
        })
        
        # Update future_df untuk iterasi berikutnya
        new_row = future_df.iloc[-1].copy()
        new_row['Tanggal'] = current_date
        new_row['Terakhir'] = next_price
        new_row['Return'] = next_return
        future_df.loc[len(future_df)] = new_row
        
        last_price = next_price
    
    # Tambahkan prediksi masa depan ke result (pastikan tidak ada duplikat)
    if len(future_predictions) > 0:
        future_df_pred = pd.DataFrame(future_predictions)
        
        # Hapus dulu tanggal yang mungkin sudah ada di result_df
        existing_future_dates = future_df_pred['Tanggal'].tolist()
        result_df = result_df[~result_df['Tanggal'].isin(existing_future_dates)].copy()
        
        # Baru append
        result_df = pd.concat([result_df, future_df_pred], ignore_index=True)
    
    result_df = result_df.sort_values('Tanggal').drop_duplicates(subset=['Tanggal'], keep='last').reset_index(drop=True)
    
    print(f"   ✅ Total baris final: {len(result_df)}")
    
    return result_df

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
    return ['Terakhir', 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30',
            'RSI', 'MACD', 'MACD_signal',
            'Terakhir_lag_1', 'Return_lag_1',
            'Terakhir_lag_2', 'Return_lag_2',
            'Terakhir_lag_3', 'Return_lag_3',
            'Terakhir_lag_4', 'Return_lag_4',
            'Terakhir_lag_5', 'Return_lag_5']