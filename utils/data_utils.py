# ===============================
# UTILITAS
# ===============================

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
# Guard imports to provide clearer instructions if a dependency is missing
try:
    import yfinance as yf
except ImportError:
    print("Error: paket 'yfinance' tidak ditemukan. Install dependensi dengan: py -3 -m pip install yfinance pandas numpy joblib xgboost openpyxl")
    sys.exit(1)

TICKER = "^JKSE"

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
    wib = ZoneInfo("Asia/Jakarta")
    today = datetime.now(wib).date()  # tanggal saat ini di WIB
    start_date = "1993-01-01"
    print(f"📈 Mengambil data IHSG terbaru tanggal {today} dari Yahoo Finance...")
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

