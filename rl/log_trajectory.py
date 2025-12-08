import sys
sys.path.append("/content/rl-liquidity-project")

import csv
import os

import numpy as np
from stable_baselines3 import PPO
from env.liquidity_env import LiquidityEnv


def log_single_episode(model, env, max_steps=500):
    obs, info = env.reset()
    step = 0

    rows = []

    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        # obs = [liquidity, volatility, current_apy]
        liquidity = float(obs[0])
        volatility = float(obs[1])
        apy = float(obs[2])

        rows.append({
            "step": step,
            "liquidity": liquidity,
            "volatility": volatility,
            "apy": apy,
            "reward": float(reward),
            "action": int(action),
        })

        step += 1
        if terminated or truncated:
            break

    return rows


def main():
    # Load trained PPO model
    model_path = "rl/models/ppo_liquidity"
    model = PPO.load(model_path)

    # Create environment
    env = LiquidityEnv()

    # Run one long episode and collect data
    rows = log_single_episode(model, env, max_steps=500)

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    csv_path = "data/ppo_trajectory.csv"

    # Write CSV
    fieldnames = ["step", "liquidity", "volatility", "apy", "reward", "action"]
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} steps to {csv_path}")


if __name__ == "__main__":
    main()
