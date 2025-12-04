import sys
sys.path.append("/content/rl-liquidity-project")

import numpy as np
from stable_baselines3 import PPO
from env.liquidity_env import LiquidityEnv


def run_episode_with_model(model, env, max_steps=500):
    obs, info = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward


def rule_based_policy(obs):
    """
    Very simple baseline:
    - If liquidity < 0.4: increase APY strongly (action 4 = +20 bp)
    - If liquidity between 0.4 and 0.6: do nothing (action 2)
    - If liquidity > 0.6: decrease APY (action 1 = -10 bp)
    """
    liquidity = float(obs[0])

    if liquidity < 0.4:
        return 4  # +20 bp
    elif liquidity > 0.6:
        return 1  # -10 bp
    else:
        return 2  # 0 bp


def run_episode_with_rule(env, max_steps=500):
    obs, info = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        action = rule_based_policy(obs)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward


def main():
    # Load trained PPO model
    model_path = "rl/models/ppo_liquidity"
    model = PPO.load(model_path)

    n_episodes = 10
    rl_rewards = []
    rule_rewards = []

    for i in range(n_episodes):
        # New env for each run to avoid state carryover
        env_rl = LiquidityEnv()
        env_rule = LiquidityEnv()

        rl_total = run_episode_with_model(model, env_rl)
        rule_total = run_episode_with_rule(env_rule)

        rl_rewards.append(rl_total)
        rule_rewards.append(rule_total)

        print(f"Episode {i+1}: RL reward = {rl_total:.3f}, Rule reward = {rule_total:.3f}")

    print("=====================================")
    print(f"RL mean total reward   : {np.mean(rl_rewards):.3f} ± {np.std(rl_rewards):.3f}")
    print(f"Rule mean total reward : {np.mean(rule_rewards):.3f} ± {np.std(rule_rewards):.3f}")


if __name__ == "__main__":
    main()
