cat > dashboard/live_data.py << 'EOF'
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta

@st.cache_data(ttl=60)
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    """Yahoo Finance CSV - works with all pandas versions."""
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={int((datetime.now() - timedelta(days=7)).timestamp())}&period2={int(datetime.now().timestamp())}&interval=5m&events=history&includeAdjustedClose=true"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        # Use io.StringIO (universal across pandas versions)
        df = pd.read_csv(io.StringIO(resp.text))
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime').reset_index(drop=True)
        return df
    except Exception as e:
        raise ValueError(f"No data for {symbol}: {str(e)[:100]}")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['price'] = df['Close']
    df['return'] = df['price'].pct_change()
    df['volatility'] = df['return'].rolling(12, min_periods=3).std()
    return df
EOF
