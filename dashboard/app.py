import streamlit as st
import pandas as pd
import time
from manual_vs_rl import run_comparison

st.set_page_config(page_title="RL Liquidity Controller", layout="wide")
st.title("🤖 RL vs Manual Liquidity Controller")

st.markdown("""
Compare your manual APY adjustments against a trained **PPO agent** in real-time.
**Goal**: Maximize liquidity while minimizing volatility (reward = liquidity - volatility - 0.1×APY).
""")

# Sidebar controls
st.sidebar.header("🎮 Manual Controller")
manual_apy_change = st.sidebar.slider(
    "APY change per step (%)", -5.0, 5.0, 0.0, 0.1
)
num_steps = st.sidebar.slider("Steps to run", 100, 1000, 500, 50)
strategy = st.sidebar.selectbox(
    "Strategy", 
    ["Constant", "Increasing", "Decreasing", "Random"]
)

# Generate manual actions based on strategy
if st.sidebar.button("🚀 Run Comparison"):
    if strategy == "Constant":
        manual_actions = [manual_apy_change / 100] * num_steps
    elif strategy == "Increasing":
        manual_actions = np.linspace(-0.02, 0.02, num_steps).tolist()
    elif strategy == "Decreasing":
        manual_actions = np.linspace(0.02, -0.02, num_steps).tolist()
    else:  # Random
        manual_actions = (np.random.uniform(-0.05, 0.05, num_steps)).tolist()
    
    # Run comparison
    with st.spinner("Running RL vs Manual..."):
        df = run_comparison(manual_actions, num_steps)
    
    # Results
    st.subheader("🏆 Final Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RL Liquidity", f"{df['rl_liquidity'].iloc[-1]:.3f}")
    col2.metric("Manual Liquidity", f"{df['manual_liquidity'].iloc[-1]:.3f}")
    col3.metric("RL Mean Reward", f"{df['rl_reward'].mean():.3f}")
    col4.metric("Manual Mean Reward", f"{df['manual_reward'].mean():.3f}")
    
    # Charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**💧 Liquidity**")
        st.line_chart({
            "RL": df['rl_liquidity'],
            "Manual": df['manual_liquidity']
        })
    
    with col_right:
        st.markdown("**⚡ Volatility**")
        st.line_chart({
            "RL": df['rl_volatility'],
            "Manual": df['manual_volatility']
        })
    
    st.markdown("**📈 APY & Reward**")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart({
            "RL": df['rl_apy'] * 100,
            "Manual": df['manual_apy'] * 100
        }, y_axis_title="APY %")
    with col2:
        st.line_chart({
            "RL": df['rl_reward'],
            "Manual": df['manual_reward']
        }, y_axis_title="Reward")
    
    # Winner
    if df['rl_reward'].mean() > df['manual_reward'].mean():
        st.success("🎉 **RL Wins!** Try a better strategy.")
    else:
        st.balloons()
        st.success("🏆 **You beat RL!** Great strategy!")
