import os
import sys

# Add repo root to Python path (one level up from 'scripts')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from env.liquidity_env import LiquidityEnv

from env.liquidity_env import LiquidityEnv

def main():
    env = LiquidityEnv()
    obs, info = env.reset()
    print("Initial observation:", obs)

    for _ in range(10):
        # Take a random action just to see the dynamics
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"  Action={action}, Reward={reward:.4f}")
        if terminated or truncated:
            print("Episode finished early.")
            break

if __name__ == "__main__":
    main()
