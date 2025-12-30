
import streamlit as st
import numpy as np
import pandas as pd
from manual_vs_rl import run_comparison
#from live_data import fetch_intraday, compute_features

st.set_page_config(page_title="RL Liquidity Controller", layout="wide")
st.title("🤖 RL vs Manual Liquidity Controller")

tab_rl, tab_live = st.tabs(["RL vs Manual", "Live Stock (read‑only)"])

# ---------------------------------------------------------------------
# TAB 1: RL vs Manual (existing)
# ---------------------------------------------------------------------
with tab_rl:
    st.markdown("""
Compare your manual APY adjustments against a trained **PPO agent** in real-time.
Goal: maximize **liquidity** while minimizing **volatility** (reward = liquidity - volatility - 0.1×APY).
""")

    st.sidebar.header("Manual Controller")
    manual_apy_change = st.sidebar.slider(
        "APY change per step (%)", -5.0, 5.0, 0.0, 0.1
    )
    num_steps = st.sidebar.slider("Steps to run", 100, 1000, 300, 50)
    strategy = st.sidebar.selectbox(
        "Strategy",
        ["Constant", "Increasing", "Decreasing", "Random"],
    )
    run_button = st.sidebar.button("🚀 Run Comparison", use_container_width=True)

    if run_button:
        # Build manual actions according to strategy
        if strategy == "Constant":
            manual_actions = [manual_apy_change / 100.0] * num_steps
        elif strategy == "Increasing":
            manual_actions = np.linspace(-0.02, 0.02, num_steps).tolist()
        elif strategy == "Decreasing":
            manual_actions = np.linspace(0.02, -0.02, num_steps).tolist()
        else:  # "Random"
            manual_actions = np.random.uniform(-0.05, 0.05, num_steps).tolist()

        with st.spinner(f"Running {num_steps} steps..."):
            df = run_comparison(manual_actions, num_steps)

        if len(df) == 0:
            st.error("No steps were recorded. Check the environment or model configuration.")
        else:
            # Summary metrics
            st.subheader("🏆 Final Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RL Liquidity", f"{df['rl_liquidity'].iloc[-1]:.3f}")
            col2.metric("Manual Liquidity", f"{df['manual_liquidity'].iloc[-1]:.3f}")
            col3.metric("RL Mean Reward", f"{df['rl_reward'].mean():.3f}")
            col4.metric("Manual Mean Reward", f"{df['manual_reward'].mean():.3f}")

            # Liquidity & Volatility
            col_lv1, col_lv2 = st.columns(2)
            with col_lv1:
                st.markdown("💧 **Liquidity**")
                st.line_chart(
                    {
                        "Manual": df["manual_liquidity"],
                        "RL": df["rl_liquidity"],
                    }
                )
            with col_lv2:
                st.markdown("⚡ **Volatility**")
                st.line_chart(
                    {
                        "Manual": df["manual_volatility"],
                        "RL": df["rl_volatility"],
                    }
                )

            # APY & Reward
            st.markdown("📈 **APY & Reward**")
            col_ar1, col_ar2 = st.columns(2)
            with col_ar1:
                st.markdown("**APY (%)**")
                st.line_chart(
                    {
                        "Manual": df["manual_apy"] * 100.0,
                        "RL": df["rl_apy"] * 100.0,
                    }
                )
            with col_ar2:
                st.markdown("**Reward**")
                st.line_chart(
                    {
                        "Manual": df["manual_reward"],
                        "RL": df["rl_reward"],
                    }
                )

            rl_mean = df["rl_reward"].mean()
            manual_mean = df["manual_reward"].mean()
            if rl_mean > manual_mean:
                st.error(
                    f"🤖 RL wins (RL: {rl_mean:.3f} > Manual: {manual_mean:.3f}). "
                    "Try a different manual strategy!"
                )
            else:
                st.balloons()
                st.success(
                    f"🎉 You beat RL (Manual: {manual_mean:.3f} > RL: {rl_mean:.3f})!"
                )
    else:
        st.info("Set your strategy on the left and click **Run Comparison** to start.")

# ---------------------------------------------------------------------
# TAB 2: Live Stock (read‑only)
# ---------------------------------------------------------------------
with tab_live:
    st.markdown("""
    **Live Stock Demo** (placeholder - working on API integration)
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        symbol = st.text_input("Stock symbol", value="AAPL")
        if st.button("📈 Show Demo"):
            pass
    
    with col2:
        st.info("✅ RL vs Manual tab works perfectly!")
        st.info("🔄 Live stock data coming soon...")
        
        # Fake demo charts
        steps = np.linspace(0, 100, 100)
        price = 150 + 5 * np.sin(steps / 10) + np.random.normal(0, 1, 100)
        vol = 0.02 + 0.01 * np.abs(np.sin(steps / 5)) + np.random.normal(0, 0.002, 100)
        
        st.subheader("💰 Price")
        st.line_chart(price)
        
        st.subheader("📊 Volatility")
        st.line_chart(vol)
