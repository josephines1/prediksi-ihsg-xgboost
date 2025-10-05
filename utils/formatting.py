import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from datetime import timedelta
import holidays

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

def save_to_excel(forecast_df, model_input_df, df_all_raw, output_path, skip_holidays=True):
    """
    Save historis + prediksi ke satu Excel sheet (Sheet1) dan simpan model_input_df ke sheet terpisah.
    - forecast_df: harus punya kolom 'Predicted_Price' (float) dan, bila ada, 'Tanggal' (datetime)
    - df_all_raw: original raw historical dataframe (harus punya kolom 'Tanggal', 'Terakhir', dll.)
    - output_path: path file .xlsx
    - skip_holidays: kalau True akan melewati libur nasional Indonesia ketika membuat tanggal prediksi baru
    """
    # helper formatting (Euro / lokal style: '.' ribuan, ',' desimal)
    def fmt_price_eu(x):
        try:
            if pd.isna(x):
                return ""
            v = float(x)
        except Exception:
            return ""
        v = round(v, 2)
        s = f"{v:,.2f}"            # '1,234.56'
        s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
        return s

    def fmt_volume_eu(x):
        if pd.isna(x):
            return ""
        try:
            v = float(x)
        except Exception:
            return ""
        if abs(v) >= 1e9:
            out = v / 1e9
            s = f"{out:.2f}".replace('.', ',') + 'B'
        elif abs(v) >= 1e6:
            out = v / 1e6
            s = f"{out:.2f}".replace('.', ',') + 'M'
        else:
            return fmt_price_eu(v)
        return s

    # --- Prepare historical dataframe ---
    df_hist = df_all_raw.copy()
    # parse tanggal dd/mm/YYYY (dayfirst=True) untuk menghindari kebalikan hari/bulan
    df_hist['Tanggal'] = pd.to_datetime(df_hist.get('Tanggal'), dayfirst=True, errors='coerce')
    df_hist = df_hist.dropna(subset=['Tanggal']).sort_values('Tanggal').reset_index(drop=True)

    # last historical date
    if df_hist.empty:
        raise RuntimeError("Data historis kosong setelah parsing tanggal.")
    last_hist_date = df_hist['Tanggal'].max().normalize()

    # --- Prepare forecast_df dates ---
    fc = forecast_df.copy().reset_index(drop=True)
    # normalisasi tanggal di forecast_df (jika ada)
    if 'Tanggal' in fc.columns:
        fc['Tanggal'] = pd.to_datetime(fc['Tanggal'], dayfirst=True, errors='coerce')

    # if forecast_df dates are missing or earlier than last_hist_date -> regenerate sensible dates
    need_regen = False
    if fc['Tanggal'].isna().all():
        need_regen = True
    else:
        # if any forecast date <= last_hist_date -> regen (we want preds after hist)
        if fc['Tanggal'].min() <= last_hist_date:
            need_regen = True

    if need_regen:
        # generate business-like dates after last_hist_date
        n = len(fc)
        dates = []
        candidate = last_hist_date + timedelta(days=1)
        id_holidays = holidays.Indonesia() if skip_holidays else set()
        while len(dates) < n:
            # skip weekends
            if candidate.weekday() >= 5:
                candidate += timedelta(days=1)
                continue
            # skip holidays if requested
            if skip_holidays and candidate in id_holidays:
                candidate += timedelta(days=1)
                continue
            dates.append(candidate)
            candidate += timedelta(days=1)
        fc['Tanggal'] = pd.to_datetime(dates)

    # filter forecast rows strictly after last_hist_date
    fc = fc[fc['Tanggal'] > last_hist_date].reset_index(drop=True)

    # --- Build output rows: historis then prediksi (jangan duplicate tanggal) ---
    out_rows = []
    out_cols = ['Tanggal', 'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%', 'Status']

    # historis rows
    for _, r in df_hist.iterrows():
        out_rows.append({
            'Tanggal': r['Tanggal'].strftime('%d/%m/%Y'),
            'Terakhir': fmt_price_eu(r.get('Terakhir', np.nan)),
            'Pembukaan': fmt_price_eu(r.get('Pembukaan', np.nan)),
            'Tertinggi': fmt_price_eu(r.get('Tertinggi', np.nan)),
            'Terendah': fmt_price_eu(r.get('Terendah', np.nan)),
            'Vol.': fmt_volume_eu(r.get('Vol.', np.nan)),
            'Perubahan%': (f"{r['Perubahan%']:.2f}".replace('.', ',') + '%') if pd.notna(r.get('Perubahan%')) else '',
            'Status': 'Historis'
        })

    # prediksi rows (hanya tanggal > last_hist_date)
    for _, r in fc.iterrows():
        price_val = r.get('Predicted_Price', np.nan)
        out_rows.append({
            'Tanggal': pd.to_datetime(r['Tanggal']).strftime('%d/%m/%Y'),
            'Terakhir': fmt_price_eu(price_val),
            'Pembukaan': '',
            'Tertinggi': '',
            'Terendah': '',
            'Vol.': '',
            'Perubahan%': '',
            'Status': 'Prediksi'
        })

    out_df = pd.DataFrame(out_rows, columns=out_cols)

    # terakhir: pastikan terurut (historis dulu berdasarkan tanggal), kemudian prediksi
    out_df['__tgl_sort'] = pd.to_datetime(out_df['Tanggal'], dayfirst=True, errors='coerce')
    out_df = out_df.sort_values('__tgl_sort').drop(columns='__tgl_sort').reset_index(drop=True)

    # --- Simpan ke Excel (Sheet1 + sheet model_input untuk debug) ---
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
            out_df.to_excel(writer, index=False, sheet_name='Sheet1')
            # tulis juga model_input_df (jika tersedia) untuk referensi — convert datetimes to strings
            try:
                mid = model_input_df.copy()
                if 'Tanggal' in mid.columns:
                    mid['Tanggal'] = pd.to_datetime(mid['Tanggal'], errors='coerce').dt.strftime('%d/%m/%Y')
                mid.to_excel(writer, index=False, sheet_name='model_input')
            except Exception:
                # jika gagal menulis model_input, lanjutkan tanpa crash
                pass
        print(f"💾 File tersimpan: {output_path} (Sheet1 + model_input)")
    except PermissionError:
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        fallback = os.path.join(os.path.dirname(__file__), f"ihsg_forecast_fallback_{ts}.xlsx")
        with pd.ExcelWriter(fallback, engine='openpyxl', mode='w') as writer:
            out_df.to_excel(writer, index=False, sheet_name='Sheet1')
        print(f"⚠️ File terkunci, disimpan ke fallback: {fallback}")