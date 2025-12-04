import sys
sys.path.append("/content/rl-liquidity-project")

import numpy as np
from stable_baselines3 import PPO
from env.liquidity_env import LiquidityEnv


def run_single_episode(model, env, max_steps=500, render=False):
    obs, info = env.reset()
    total_reward = 0.0

    liquidity_history = []
    volatility_history = []
    apy_history = []
    reward_history = []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        total_reward += reward

        # obs = [liquidity, volatility, current_apy]
        liquidity_history.append(float(obs[0]))
        volatility_history.append(float(obs[1]))
        apy_history.append(float(obs[2]))
        reward_history.append(float(reward))

        if render:
            env.render()

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "liquidity": liquidity_history,
        "volatility": volatility_history,
        "apy": apy_history,
        "rewards": reward_history,
    }


def main():
    # Load trained PPO model
    model_path = "rl/models/ppo_liquidity"
    model = PPO.load(model_path)

    # Create a fresh environment
    env = LiquidityEnv()

    # Run multiple evaluation episodes
    n_episodes = 5
    episode_rewards = []

    for i in range(n_episodes):
        result = run_single_episode(model, env, render=False)
        episode_rewards.append(result["total_reward"])
        print(f"Episode {i+1}: total_reward = {result['total_reward']:.3f}")

    print("===================================")
    print(f"Mean total reward over {n_episodes} episodes: {np.mean(episode_rewards):.3f}")
    print(f"Std of total reward: {np.std(episode_rewards):.3f}")


if __name__ == "__main__":
    main()


# this tells you whether or not the general policy is doing well
