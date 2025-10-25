#!/usr/bin/env python3
"""
build_features.py

Reads the latest processed OHLCV file from data/processed (CSV or parquet),
builds features (returns, moving averages, RSI, volatility, time features),
ensures no lookahead, and writes a features parquet file to data/processed/.

Usage:
    python src/features/build_features.py
    python src/features/build_features.py --input data/processed/processed_ETHUSDT_1h_....parquet
    python src/features/build_features.py --output data/processed/eth_features_1h.parquet
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

# -----------------------
# Utils / Indicators
# -----------------------
def compute_return(df, shift=1):
    # simple percent return: (close_t / close_{t-shift} - 1)
    return df['close'].pct_change(periods=shift)

def compute_sma(df, window):
    return df['close'].rolling(window=window, min_periods=window).mean()

def compute_rsi(df, window=14):
    # classic RSI (Wilder's smoothing approximated with simple moving averages)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use exponential weighted moving average for more stable RSI (Wilder's method-like)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_volatility(df, window=24):
    # volatility = rolling std of returns over window (not annualized here)
    ret = df['close'].pct_change()
    vol = ret.rolling(window=window, min_periods=window).std()
    return vol

# -----------------------
# Main pipeline
# -----------------------
def load_latest_processed(input_path: Path = None):
    processed_dir = Path('data/processed')
    if input_path:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p
    # choose latest processed file by modified time that starts with processed_
    files = sorted(processed_dir.glob('processed_*'), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No processed_* files found in {processed_dir}")
    return files[-1]

def build_features(df: pd.DataFrame, verbose=True):
    # Ensure columns exist
    expected = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Input dataframe missing expected columns. Found: {df.columns.tolist()}")

    # Ensure timestamp is datetime and sorted
    if not (is_datetime64_any_dtype(df['timestamp']) or is_datetime64tz_dtype(df['timestamp'])):
        # handle integer unix-ms timestamps or parse strings
        if pd.api.types.is_integer_dtype(df['timestamp']) or pd.api.types.is_float_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms', utc=True, errors='coerce')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Core features (no lookahead)
    df['ret1'] = compute_return(df, shift=1)
    df['sma_12'] = compute_sma(df, window=12)
    df['rsi_14'] = compute_rsi(df, window=14)
    df['volatility_24'] = compute_volatility(df, window=24)

    # Time features (derived from timestamp only)
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Optionally: target column for direction (1 if next period return > 0 else 0)
    df['ret_next'] = df['close'].pct_change(periods=-1)  # next period return (note: lookahead until we shift)
    # We want target using only t data during training; store shifted target where index t holds next return
    df['target_dir'] = (df['ret_next'] > 0).astype(int)

    # Drop columns we do not need as features (keep target and features)
    # Compute warm-up: the maximum rolling window used
    warmup = max(12, 14, 24)  # windows used: sma_12, rsi_14, volatility_24
    # drop last row because target uses next period (ret_next is NaN for last row)
    df = df.iloc[warmup:-1].copy()  # drop initial warmup rows and last row (no next return)
    df.reset_index(drop=True, inplace=True)

    # Final check: no NaNs in features requested (after warmup)
    feature_cols = ['ret1', 'sma_12', 'rsi_14', 'volatility_24', 'hour', 'dayofweek', 'month']
    missing = df[feature_cols].isna().sum()
    if verbose:
        print("Feature NaNs after warmup (should be 0):")
        print(missing.to_dict())
        print(f"Rows after drop: {len(df)}")

    if missing.sum() > 0:
        # As a safety: forward-fill/backfill any remaining NaNs (shouldn't be necessary)
        df[feature_cols] = df[feature_cols].ffill().bfill()

    return df

def main(args):
    input_path = load_latest_processed(args.input) if not args.input else Path(args.input)
    print(f"Using input file: {input_path}")
    # read CSV or parquet
    if input_path.suffix.lower() in ['.csv', '.txt']:
        df = pd.read_csv(input_path)
        # try to coerce timestamp if numeric string
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.datetime64):
            # if timestamp looks like unix ms integers, keep numeric; else parse
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            except Exception:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    else:
        df = pd.read_parquet(input_path)

    if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'timestamp'})

    features_df = build_features(df, verbose=not args.quiet)

    # prepare output
    out_dir = Path(args.output_dir or 'data/processed')
    out_dir.mkdir(parents=True, exist_ok=True)
    # create output filename from input
    base = input_path.stem.replace('processed_', '').replace('.parquet', '')
    # Add symbol/timeframe detection fallback - use timestamp now if can't derive
    timestamp_suffix = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_name = args.output or f"{base}_features_{timestamp_suffix}.parquet"
    out_path = out_dir / out_name

    features_df.to_parquet(out_path, index=False)
    print(f"✅ Features saved to: {out_path}")
    # Acceptance: check columns exist
    required_cols = ['ret1', 'sma_12', 'rsi_14', 'volatility_24', 'target_dir']
    missing_required = [c for c in required_cols if c not in features_df.columns]
    if missing_required:
        raise RuntimeError(f"Missing required feature columns: {missing_required}")
    print("Acceptance check passed: required columns present and NaNs handled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features from processed OHLCV data")
    parser.add_argument("--input", "-i", help="Input processed file (CSV or parquet). If omitted, uses latest in data/processed/")
    parser.add_argument("--output", "-o", dest="output", help="Output filename (parquet) under data/processed/")
    parser.add_argument("--output-dir", dest="output_dir", help="Output directory (default data/processed/)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose printing")
    args = parser.parse_args()
    main(args)
