cat > dashboard/live_data.py << 'EOF'
import streamlit as st
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import numpy as np

@st.cache_data(ttl=60)
def fetch_stock_data(symbol: str, period_days: int = 1) -> pd.DataFrame:
    """Fetch stock data using pandas_datareader + Yahoo (no extra deps, unlimited)."""
    end = datetime.now()
    start = end - timedelta(days=period_days)
    
    try:
        df = web.DataReader(symbol, 'yahoo', start, end)
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception as e:
        raise ValueError(f"No data for {symbol}: {e}")

def compute_features(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Compute returns and rolling volatility."""
    df = df.copy()
    df['price'] = df['close']
    df['return'] = df['price'].pct_change()
    df['volatility'] = df['return'].rolling(window=window, min_periods=3).std()
    return df
EOF
