import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.liquidity_env import LiquidityEnv

def make_env():
    return LiquidityEnv()

if __name__ == "__main__":
    # Vectorized environment required by Stable-Baselines3 PPO
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1)

    # Train for 200,000 timesteps (adjust as needed)
    model.learn(total_timesteps=200000)

    # Ensure models folder exists
    os.makedirs("rl/models", exist_ok=True)

    # Save the trained model
    model.save("rl/models/ppo_liquidity")
