import os
import glob
import json
import pandas as pd
from datetime import datetime

# -------------------------------
# Config
# -------------------------------
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'


def calculate_indicators(df):
    """Add simple technical indicators like moving averages and RSI."""
    # 10 and 30 period simple moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df


def _to_datetime_utc(val):
    """Convert timestamp-like value to pandas.Timestamp (UTC)."""
    if pd.isna(val):
        return pd.NaT
    # numeric (ms or seconds)
    if isinstance(val, (int, float)):
        if val > 1e12:
            return pd.to_datetime(int(val), unit='ms', utc=True)
        if val < 1e10:
            return pd.to_datetime(int(val * 1000), unit='ms', utc=True)
        return pd.to_datetime(int(val), unit='ms', utc=True)
    # string
    try:
        return pd.to_datetime(val).tz_convert('UTC') if isinstance(val, pd.DatetimeIndex) else pd.to_datetime(val, utc=True)
    except Exception:
        return pd.NaT


def get_latest_raw_file():
    """Return the most recent raw data file (parquet/csv/json) or raise."""
    patterns = [os.path.join(RAW_DIR, '*.parquet'),
                os.path.join(RAW_DIR, '*.csv'),
                os.path.join(RAW_DIR, '*.json')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No raw files found in {RAW_DIR} (searched parquet/csv/json)")
    return max(files, key=os.path.getctime)


def load_raw_file(path):
    """Load CSV/Parquet/JSON produced by fetch into a pandas DataFrame with a 'timestamp' column."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.parquet':
        df = pd.read_parquet(path)
    elif ext == '.csv':
        df = pd.read_csv(path)
    elif ext == '.json':
        # try to detect shape: list of lists (ohlcv) or list of dicts
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # list of lists -> OHLCV format
        if isinstance(data, list) and data and isinstance(data[0], list):
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        else:
            df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported raw file extension: {ext}")

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    if 'timestamp' not in (c.lower() for c in df.columns):
        # try detect common timestamp columns
        for candidate in ['time', 'date', 'datetime']:
            if candidate in cols:
                df.rename(columns={cols[candidate]: 'timestamp'}, inplace=True)
                break

    # If index is datetime, reset to column
    if isinstance(df.index, pd.DatetimeIndex) and 'timestamp' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'timestamp'})

    if 'timestamp' not in df.columns:
        raise ValueError("Loaded raw file does not contain a 'timestamp' column")

    # Convert timestamp values to pandas datetime UTC
    df['timestamp'] = df['timestamp'].apply(_to_datetime_utc)
    df = df.set_index('timestamp').sort_index()

    return df


def process_latest_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    latest_file = get_latest_raw_file()
    print(f"Processing latest raw file: {latest_file}")

    df = load_raw_file(latest_file)

    # Drop exact duplicate timestamps, keep first
    df = df[~df.index.duplicated(keep='first')]

    # Detect numeric price columns and ensure they are floats
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing rows: forward fill prices, zero-fill volume if missing
    df.sort_index(inplace=True)
    df_ffill = df.copy()

    # Forward fill numeric price columns
    price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df_ffill.columns]
    if price_cols:
        df_ffill[price_cols] = df_ffill[price_cols].ffill()

    if 'volume' in df_ffill.columns:
        df_ffill['volume'] = df_ffill['volume'].fillna(0)

    # If there are remaining NaNs, fill with forward then backward as a last resort
    df_ffill = df_ffill.fillna(method='ffill').fillna(method='bfill')

    # Calculate indicators (requires 'close')
    if 'close' not in df_ffill.columns:
        raise RuntimeError("Cannot compute indicators: 'close' column is missing after cleaning.")
    df_enriched = calculate_indicators(df_ffill)

    # Save processed outputs (CSV + parquet) with a clear name
    base = os.path.splitext(os.path.basename(latest_file))[0]
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_csv = os.path.join(PROCESSED_DIR, f"processed_{base}_{timestamp}.csv")
    out_parquet = os.path.join(PROCESSED_DIR, f"processed_{base}_{timestamp}.parquet")

    df_enriched.to_csv(out_csv, index=True)
    try:
        df_enriched.to_parquet(out_parquet, index=True)
    except Exception:
        # parquet optional; ignore failures
        pass

    print(f"Processed data saved: {out_csv} (and parquet: {out_parquet} if supported)")


if __name__ == "__main__":
    process_latest_data()
