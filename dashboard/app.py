import os
import time

import numpy as np
import pandas as pd
import streamlit as st


DATA_PATH = "data/ppo_trajectory.csv"


@st.cache_data
def load_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run rl/log_trajectory.py first.")
    df = pd.read_csv(csv_path)
    # Ensure sorted by step
    df = df.sort_values("step").reset_index(drop=True)
    return df


def render_sidebar(df: pd.DataFrame):
    st.sidebar.header("Playback controls")

    max_step = int(df["step"].max())
    total_steps = st.sidebar.slider(
        "Number of steps to play",
        min_value=50,
        max_value=max_step,
        value=min(500, max_step),
        step=10,
    )
    speed = st.sidebar.slider(
        "Playback speed (seconds per step)",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01,
    )
    loop = st.sidebar.checkbox("Loop playback", value=False)

    st.sidebar.markdown("### Plot options")
    show_table = st.sidebar.checkbox("Show raw data table", value=False)

    return total_steps, speed, loop, show_table


def render_header():
    st.set_page_config(
        page_title="RL Liquidity Controller",
        layout="wide",
    )
    st.title("RL Liquidity Controller – PPO Trajectory")
    st.markdown(
        """
This dashboard replays a single episode of a trained PPO agent controlling APY in a toy liquidity pool environment.

- **Liquidity** should increase and stabilize.
- **Volatility** should decrease as liquidity deepens.
- **APY** is adjusted by the agent and may saturate at its upper bound.
- **Reward** reflects the trade‑off between liquidity, volatility, and APY cost.
"""
    )


def render_summary(df: pd.DataFrame):
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Liquidity", f"{latest['liquidity']:.3f}")
    col2.metric("Final Volatility", f"{latest['volatility']:.3f}")
    col3.metric("Final APY", f"{latest['apy']*100:.2f}%")
    col4.metric("Mean Reward", f"{df['reward'].mean():.3f}")


def main():
    render_header()
    df = load_data()

    total_steps, speed, loop, show_table = render_sidebar(df)

    st.subheader("Episode summary")
    render_summary(df)

    if show_table:
        st.markdown("#### Raw trajectory data (head)")
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")
    st.subheader("Trajectory playback")

    # Layout: big liquidity/APY chart on left, volatility/reward on right
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Liquidity & APY over time**")
        liq_chart = st.line_chart({"step": [], "liquidity": [], "apy": []})

    with col_right:
        st.markdown("**Volatility & Reward over time**")
        vol_chart = st.line_chart({"step": [], "volatility": [], "reward": []})

    # Playback controls
    start_button = st.button("Start playback")
    placeholder_info = st.empty()

    if start_button:
        while True:
            # Clear charts before each run
            liq_chart.empty()
            vol_chart.empty()

            for t in range(total_steps):
                current = df.iloc[: t + 1]

                liq_chart.add_rows(
                    {
                        "step": [current["step"].iloc[-1]],
                        "liquidity": [current["liquidity"].iloc[-1]],
                        "apy": [current["apy"].iloc[-1]],
                    }
                )
                vol_chart.add_rows(
                    {
                        "step": [current["step"].iloc[-1]],
                        "volatility": [current["volatility"].iloc[-1]],
                        "reward": [current["reward"].iloc[-1]],
                    }
                )

                placeholder_info.markdown(
                    f"**Step:** {int(current['step'].iloc[-1])} &nbsp;&nbsp; "
                    f"**Liquidity:** {current['liquidity'].iloc[-1]:.3f} &nbsp;&nbsp; "
                    f"**Volatility:** {current['volatility'].iloc[-1]:.3f} &nbsp;&nbsp; "
                    f"**APY:** {current['apy'].iloc[-1]*100:.2f}% &nbsp;&nbsp; "
                    f"**Reward:** {current['reward'].iloc[-1]:.3f}"
                )

                time.sleep(speed)

            if not loop:
                break


if __name__ == "__main__":
    main()
