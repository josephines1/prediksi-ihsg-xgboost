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
