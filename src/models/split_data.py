"""
split_data.py — Split features into train/val/test chronologically

Usage:
    python src/models/split_data.py
    python src/models/split_data.py --input data/processed/...features....parquet
    python src/models/split_data.py --train 0.7 --val 0.15
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

def load_features_file(input_path=None):
    processed_dir = Path("data/processed")
    if input_path:
        return Path(input_path)
    files = sorted(processed_dir.glob("*features_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No feature files found in data/processed.")
    return files[-1]

def time_split(df, train_ratio=0.7, val_ratio=0.15):
    """Split by chronological order (no shuffle)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    print(f"Splits -> train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    print("Date ranges:")
    for name, subset in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        print(f"{name}: {subset['timestamp'].iloc[0]} → {subset['timestamp'].iloc[-1]}")

    return df_train, df_val, df_test

def main(args):
    input_path = load_features_file(args.input)
    print(f"Loading features from: {input_path}")
    df = pd.read_parquet(input_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df_train, df_val, df_test = time_split(df, args.train, args.val)

    out_dir = Path(args.output_dir or "data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_suffix = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    df_train.to_parquet(out_dir / f"train_{timestamp_suffix}.parquet", index=False)
    df_val.to_parquet(out_dir / f"val_{timestamp_suffix}.parquet", index=False)
    df_test.to_parquet(out_dir / f"test_{timestamp_suffix}.parquet", index=False)

    print(f"✅ Split files saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train/val/test sets by time")
    parser.add_argument("--input", "-i", help="Input features parquet file (default: latest in data/processed)")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio (default 0.7)")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (default 0.15)")
    parser.add_argument("--output-dir", dest="output_dir", help="Output directory (default data/processed)")
    args = parser.parse_args()
    main(args)
