import sys
sys.path.append("/content/rl-liquidity-project")

from stable_baselines3 import PPO
from env.liquidity_env import LiquidityEnv


def load_trained_model(model_path: str = "rl/models/ppo_liquidity") -> PPO:
    """Load the trained PPO model from disk."""
    return PPO.load(model_path)


def run_episode_with_model(model: PPO, max_steps: int = 500):
    """Run a single episode and return per-step data."""
    env = LiquidityEnv()
    obs, info = env.reset()

    steps = []
    liquidities = []
    volatilities = []
    apys = []
    rewards = []
    actions = []

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        steps.append(t)
        liquidities.append(float(obs[0]))
        volatilities.append(float(obs[1]))
        apys.append(float(obs[2]))
        rewards.append(float(reward))
        actions.append(int(action))

        if terminated or truncated:
            break

    return {
        "step": steps,
        "liquidity": liquidities,
        "volatility": volatilities,
        "apy": apys,
        "reward": rewards,
        "action": actions,
    }
