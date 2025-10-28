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


from datetime import timedelta
import pandas as pd
import numpy as np
import holidays

def forecast_future_incremental(model, df_all, df_all_raw, existing_df, n_forecast, exclude_holidays=True):
    """
    Update incremental: hanya generate prediksi untuk data baru.
    ...
    (docstring sama seperti sebelumnya)
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
        # copy dan normalize tanggal
        result_df = existing_df.copy()
        result_df['Tanggal'] = pd.to_datetime(result_df['Tanggal'], dayfirst=True, errors='coerce')

        # siapkan df_all (source terbaru dari yfinance) dan normalisasi
        df_all_copy = df_all.copy()
        df_all_copy['Tanggal'] = pd.to_datetime(df_all_copy.get('Tanggal'), dayfirst=True, errors='coerce')
        df_all_copy = df_all_copy.sort_values('Tanggal').reset_index(drop=True)

        # 1) Update nilai historis: untuk setiap kolom historis, replace kosong/NA di result_df
        hist_cols = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%']
        # buat mapping per kolom dari df_all_copy (latest source)
        df_all_map = df_all_copy.set_index('Tanggal')

        for col in hist_cols:
            if col in df_all_copy.columns:
                # ambil mapping; jika nilai di df_all_map kosong -> ignore
                def mapper(date):
                    try:
                        v = df_all_map.at[date, col]
                        # beberapa sumber mungkin memiliki index duplicate; handle dengan .loc fallback
                        if pd.isna(v) or v == '':
                            return np.nan
                        return v
                    except KeyError:
                        # jika tidak ada langsung return existing value (tidak diubah)
                        return np.nan
                # map values from df_all_map
                mapped = result_df['Tanggal'].map(lambda d: mapper(d))
                # combine: prefer mapped (yfinance) jika ada, else gunakan existing
                result_df[col] = mapped.combine_first(result_df.get(col))

        # 2) Pastikan format Vol. tetap sebagai string/formatted jika ada nilai numeric dari df_all
        # (opsional: kamu bisa reformat Vol. di sini jika perlu)
        # result_df['Vol.'] = result_df['Vol.'].apply(lambda x: fmt_volume(x) if pd.notna(x) else '')

        # 3) Hitung last_hist_date dari sumber yfinance (df_all_copy) => lebih stabil
        df_valid = df_all_copy[(df_all_copy['Tanggal'].notna()) & (df_all_copy['Terakhir'].notna())]
        if not df_valid.empty:
            last_hist_date = df_valid['Tanggal'].max()
        else:
            # fallback ke result_df kalau df_all kosong
            hist_rows = result_df[result_df['Terakhir'].notna() & (result_df['Terakhir'] != '')]
            if len(hist_rows) > 0:
                last_hist_date = hist_rows['Tanggal'].max()
            else:
                last_hist_date = result_df['Tanggal'].min()

        print(f"   📅 Tanggal terakhir historis (dari sumber): {last_hist_date.strftime('%d/%m/%Y')}")

        # 4) Hapus prediksi masa depan yang lama (tanggal > last_hist_date)
        #    tapi jangan hapus baris historis yang baru saja diupdate
        #    (kita akan menambahkan prediksi baru nanti)
        result_df = result_df[result_df['Tanggal'] <= last_hist_date].copy()
        print(f"   🗑️ Hapus prediksi lama, tersisa {len(result_df)} baris sampai {last_hist_date.strftime('%d/%m/%Y')}")

        # 5) Cari baris dalam result_df yang masih belum punya 'Terakhir (Prediksi)' untuk diupdate
        mask_empty = result_df['Terakhir (Prediksi)'].isna() | (result_df['Terakhir (Prediksi)'] == '') | (result_df['Terakhir (Prediksi)'] == 0)
        rows_need_prediction = result_df[mask_empty].copy()
        print(f"   🎯 {len(rows_need_prediction)} baris perlu diupdate prediksinya (existing portion)")

        # Generate prediksi untuk baris yang kosong (sama seperti sebelumnya)
        if len(rows_need_prediction) > 0:
            # buat df_all_copy yang berisi fitur model (expected_features) untuk matching
            df_all_feat = df_all_copy.copy()
            # pastikan semua expected_features ada (jika tidak ada akan kena exception di predict)
            for idx, row in rows_need_prediction.iterrows():
                tgl = row['Tanggal']
                matching = df_all_feat[df_all_feat['Tanggal'] == tgl]
                if len(matching) > 0:
                    try:
                        X = matching[expected_features]
                        pred_return = float(model.predict(X)[0])
                        pred_price = matching['Terakhir'].iloc[0] * (1 + pred_return)
                        result_df.loc[idx, 'Terakhir (Prediksi)'] = pred_price
                        print(f"      ✓ Updated Prediksi {tgl.strftime('%d/%m/%Y')}: {pred_price:.2f}")
                    except Exception as e:
                        # fallback: kalau fitur lag belum lengkap, pakai actual jika ada
                        try:
                            actual_price = matching['Terakhir'].iloc[0]
                            if pd.notna(actual_price):
                                result_df.loc[idx, 'Terakhir (Prediksi)'] = actual_price
                                print(f"      ⚠️ {tgl.strftime('%d/%m/%Y')}: Pakai nilai aktual (lag incomplete)")
                        except:
                            print(f"      ⚠️ Skip {tgl.strftime('%d/%m/%Y')}: {str(e)[:50]}")
                        continue

    else:
        # Kalau file existing tidak ada, buat dari awal
        print("   ⚠️ File existing tidak ada, buat dari awal...")
        result_df = pd.DataFrame(columns=['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%', 'Terakhir (Prediksi)'])
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
        last_hist_date = result_df['Tanggal'].max()

    # === Generate prediksi n_forecast hari ke depan ===
    print(f"   🔮 Membuat prediksi {n_forecast} hari ke depan...")

    # Siapkan df_sorted dari df_all terbaru (source data)
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

        # Ambil fitur terakhir untuk prediksi (ingat: future_df harus punya expected_features)
        # Jika expected_features tidak ada sepenuhnya, model.predict akan error -> fallback handled below
        try:
            X_last = future_df[expected_features].iloc[[-1]]
            next_return = float(model.predict(X_last)[0])
            next_price = last_price * (1 + next_return)
        except Exception as e:
            # fallback: jika gagal prediksi karena fitur tidak lengkap, gunakan last_price (no-change)
            next_return = 0.0
            next_price = last_price
            print(f"      ⚠️ Prediksi iterasi gagal (fallback no-change): {str(e)[:80]}")

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

        # Update future_df untuk iterasi berikutnya: tambahkan baris "prediksi" sehingga lag bisa dipakai
        new_row = future_df.iloc[-1].copy()
        new_row['Tanggal'] = current_date
        new_row['Terakhir'] = next_price
        # Jika model memerlukan kolom Return, tambahkan
        new_row['Return'] = next_return
        future_df.loc[len(future_df)] = new_row

        last_price = next_price

    # Tambahkan prediksi masa depan ke result (pastikan tidak ada duplikat)
    if len(future_predictions) > 0:
        future_df_pred = pd.DataFrame(future_predictions)
        existing_future_dates = future_df_pred['Tanggal'].tolist()
        result_df = result_df[~result_df['Tanggal'].isin(existing_future_dates)].copy()
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