
import os
import time
import requests
import pandas as pd
import streamlit as st

ALPHA_URL = "https://www.alphavantage.co/query"


@st.cache_data(ttl=60)
def fetch_intraday(symbol: str, interval: str = "1min") -> pd.DataFrame:
    """
    Fetch latest intraday OHLCV data for a stock symbol.
    Cached for 60 seconds to avoid hitting free-tier rate limits.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY", None)
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not set in environment or Streamlit secrets.")

    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": "compact",
        "datatype": "json",
    }

    resp = requests.get(ALPHA_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    key = f"Time Series ({interval})"
    if key not in data:
        raise RuntimeError(f"Unexpected response from Alpha Vantage: {list(data.keys())}")

    ts = data[key]  # dict[time -> {open, high, low, close, volume}]
    df = (
        pd.DataFrame.from_dict(ts, orient="index")
        .rename(columns=lambda c: c.split(". ")[-1])  # "1. open" -> "open"
        .astype(float)
        .sort_index()
    )
    df.index = pd.to_datetime(df.index)
    df = df.rename_axis("timestamp").reset_index()
    return df


def compute_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add simple features: returns, rolling volatility.
    """
    df = df.copy()
    df["price"] = df["close"]
    df["return"] = df["price"].pct_change()
    df["volatility"] = df["return"].rolling(window).std()
    return df
