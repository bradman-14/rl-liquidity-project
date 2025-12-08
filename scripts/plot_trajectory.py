import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = "data/ppo_trajectory.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run rl/log_trajectory.py first.")

    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Quick info
    print(df.head())
    print(f"Loaded {len(df)} steps from {csv_path}")

    # Create a 2x2 grid of plots: liquidity, volatility, apy, reward
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO LiquidityEnv Trajectory")

    # Liquidity over time
    axes[0, 0].plot(df["step"], df["liquidity"])
    axes[0, 0].set_title("Liquidity vs Step")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Liquidity")

    # Volatility over time
    axes[0, 1].plot(df["step"], df["volatility"])
    axes[0, 1].set_title("Volatility vs Step")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Volatility")

    # APY over time
    axes[1, 0].plot(df["step"], df["apy"])
    axes[1, 0].set_title("APY vs Step")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("APY")

    # Reward over time
    axes[1, 1].plot(df["step"], df["reward"])
    axes[1, 1].set_title("Reward vs Step")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Reward")

    plt.tight_layout()
  
    plt.show()


if __name__ == "__main__":
    main()
