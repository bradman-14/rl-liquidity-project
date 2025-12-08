import sys
sys.path.append("/content/rl-liquidity-project")

import os
import pandas as pd

from rl.episode_runner import load_trained_model, run_episode_with_model


def main():
    # Load trained PPO model
    model = load_trained_model("rl/models/ppo_liquidity")

    # Run one long episode and collect data as dict of lists
    data = run_episode_with_model(model, max_steps=500)

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    csv_path = "data/ppo_trajectory.csv"

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    print(f"Saved {len(df)} steps to {csv_path}")


if __name__ == "__main__":
    main()
