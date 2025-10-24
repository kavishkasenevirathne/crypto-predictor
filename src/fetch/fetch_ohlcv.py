import os
import time
import math
import json
import argparse
from datetime import datetime
import ccxt
import pandas as pd

def _to_ms(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        # assume milliseconds if large
        if ts > 1e12:
            return int(ts)
        # assume seconds
        if ts < 1e10:
            return int(ts * 1000)
        return int(ts)
    # string provided
    try:
        dt = pd.to_datetime(ts)
        return int(dt.timestamp() * 1000)
    except Exception:
        raise ValueError(f"cannot parse since: {ts}")

def _timeframe_to_pandas_freq(tf):
    # simple mapping for common intervals
    if tf.endswith('m'):
        n = int(tf[:-1])
        return f"{n}T"
    if tf.endswith('h'):
        n = int(tf[:-1])
        return f"{n}H"
    if tf.endswith('d'):
        n = int(tf[:-1])
        return f"{n}D"
    # fallback to 1 minute
    return "1T"

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _append_manifest(manifest_path, entry):
    header = ['created_utc', 'filename', 'symbol', 'timeframe', 'start_utc', 'end_utc', 'rows', 'source']
    exists = os.path.exists(manifest_path)
    _ensure_dir(os.path.dirname(manifest_path) or '.')
    with open(manifest_path, 'a', encoding='utf-8') as f:
        if not exists:
            f.write(','.join(header) + '\n')
        f.write(','.join(str(entry.get(h, "")) for h in header) + '\n')

def fetch_ohlcv(
    exchange_name,
    symbol,
    timeframe='1h',
    since=None,
    limit=None,
    output_dir='data/raw',
    manifest_path='data/manifest.csv',
    per_request_limit=None,
    max_retries=5,
    backoff_factor=1.5,
):
    """
    Fetch historical OHLCV with pagination, retries, rate-limit compliance,
    saves parquet + raw json, and records a manifest entry.
    Returns pandas.DataFrame indexed by timestamp (UTC).
    """
    _ensure_dir(output_dir)

    # initialize exchange
    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls({'enableRateLimit': True})
    exchange.verbose = False

    # determine per-request limit
    if per_request_limit is None:
        # many exchanges use 500/1000; let user override via per_request_limit
        per_request_limit = 1000 if not hasattr(exchange, 'has') else 1000
    if limit is not None:
        total_to_fetch = int(limit)
    else:
        total_to_fetch = None

    since_ms = _to_ms(since) if since is not None else None

    all_rows = []
    fetched = 0
    current_since = since_ms

    while True:
        # respect exchange rate limit
        sleep_secs = (getattr(exchange, 'rateLimit', 0) or 0) / 1000.0
        if sleep_secs > 0:
            time.sleep(sleep_secs)

        retries = 0
        backoff = 1.0
        last_error = None
        while retries <= max_retries:
            try:
                call_limit = per_request_limit
                # If total limit provided, reduce per-request accordingly
                if total_to_fetch is not None:
                    remaining = total_to_fetch - fetched
                    if remaining <= 0:
                        break
                    call_limit = min(call_limit, remaining)
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=call_limit)
                break
            except ccxt.NetworkError as e:
                last_error = e
                retries += 1
                time.sleep(backoff)
                backoff *= backoff_factor
                continue
            except ccxt.ExchangeError as e:
                # some exchanges return rate limit errors; backoff and retry
                last_error = e
                retries += 1
                time.sleep(backoff)
                backoff *= backoff_factor
                continue
            except Exception as e:
                last_error = e
                retries += 1
                time.sleep(backoff)
                backoff *= backoff_factor
                continue

        else:
            # exhausted retries
            raise RuntimeError(f"Failed fetching OHLCV after {max_retries} retries. Last error: {last_error}")

        if not ohlcv:
            break

        all_rows.extend(ohlcv)
        fetched += len(ohlcv)

        # if fewer than requested returned -> likely no more data
        if len(ohlcv) < call_limit:
            break

        # prepare next since: last timestamp + 1ms
        last_ts = ohlcv[-1][0]
        next_since = int(last_ts) + 1
        if current_since is not None and next_since <= current_since:
            # no progress
            break
        current_since = next_since

        # if total_to_fetch reached
        if total_to_fetch is not None and fetched >= total_to_fetch:
            break

    if not all_rows:
        return pd.DataFrame(columns=['open','high','low','close','volume']).set_index(pd.DatetimeIndex([], name='timestamp'))

    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # convert timestamp ms -> UTC datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    # Ensure continuous index for timeframe
    freq = _timeframe_to_pandas_freq(timeframe)
    start = df.index[0]
    end = df.index[-1]
    full_index = pd.date_range(start=start, end=end, freq=freq, tz='UTC')
    df = df.reindex(full_index)
    df.index.name = 'timestamp'  # keep name

    # If user provided a total limit smaller than fetched, trim to most recent 'limit' rows
    if total_to_fetch is not None:
        df = df[-total_to_fetch:]

    # prepare filenames
    start_str = start.strftime('%Y%m%dT%H%M%SZ')
    end_str = end.strftime('%Y%m%dT%H%M%SZ')
    safe_symbol = symbol.replace('/', '').replace('-', '')
    base_name = f"{safe_symbol}_{timeframe}_{start_str}_{end_str}"
    parquet_path = os.path.join(output_dir, base_name + ".parquet")
    json_path = os.path.join(output_dir, base_name + ".json")

    # Save parquet (use pyarrow if available)
    try:
        df.to_parquet(parquet_path, engine='pyarrow', index=True)
    except Exception:
        # fallback to pandas default
        df.to_parquet(parquet_path, index=True)

    # Save raw json of fetched rows (timestamps in ms)
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_rows, f, ensure_ascii=False)
    except Exception:
        pass

    # Append manifest entry
    manifest_entry = {
        'created_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'filename': os.path.relpath(parquet_path),
        'symbol': symbol,
        'timeframe': timeframe,
        'start_utc': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'end_utc': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'rows': len(df),
        'source': exchange_name
    }
    _append_manifest(manifest_path, manifest_entry)

    return df

def _cli():
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV and save to data/raw")
    parser.add_argument("--exchange", default="binance", help="ccxt exchange id (default: binance)")
    parser.add_argument("--symbol", required=True, help="Trading pair symbol e.g. BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="Timeframe, e.g. 1m, 1h, 1d")
    parser.add_argument("--since", default=None, help="Start date (ISO) or timestamp (e.g. 2023-01-01)")
    parser.add_argument("--limit", type=int, default=None, help="Total number of candles to fetch (optional)")
    parser.add_argument("--output-dir", default="data/raw", help="Directory to save raw files")
    parser.add_argument("--manifest", default="data/manifest.csv", help="Manifest CSV path")
    args = parser.parse_args()

    since = args.since
    if since is not None:
        # allow simple date like YYYY-MM-DD
        try:
            pd.to_datetime(since)
        except Exception:
            raise SystemExit("Cannot parse --since. Use ISO date like 2023-01-01 or timestamp.")

    df = fetch_ohlcv(
        exchange_name=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=since,
        limit=args.limit,
        output_dir=args.output_dir,
        manifest_path=args.manifest
    )
    if df is None:
        print("No data fetched.")
    else:
        print(f"Saved data to {args.output_dir}. Rows: {len(df)}")
        print(df.head())

if __name__ == "__main__":
    _cli()