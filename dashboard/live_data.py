cat > dashboard/live_data.py << 'EOF'
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@st.cache_data(ttl=30)
def fetch_stock_data(symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance - unlimited free calls, no API key."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data for {symbol}. Try AAPL, MSFT, GOOGL, TSLA")
    
    df = df.reset_index()
    df.columns = [col.lower() for col in df.columns]
    return df

def compute_features(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Compute returns and rolling volatility."""
    df = df.copy()
    df['price'] = df['close']
    df['return'] = df['price'].pct_change()
    df['volatility'] = df['return'].rolling(window=window, min_periods=3).std()
    return df

EOF
