import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env import LiquidityEnv


@st.cache_resource
def load_model():
    # model file lives at rl/models/ppo_liquidity.zip
    return PPO.load("rl/models/ppo_liquidity.zip")
    

def _reset_env(env):
    """Handle both Gym old and new reset API."""
    out = env.reset()
    if isinstance(out, tuple):
        obs, info = out
    else:
        obs, info = out, {}
    return obs, info


def _step_env(env, action):
    """Handle both Gym old and new step API."""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def run_comparison(manual_actions, num_steps=500):
    """Run RL vs Manual trajectories and return a comparison DataFrame."""
    env = LiquidityEnv()
    model = load_model()

    rl_next_obs = []
    rl_rewards = []
    manual_next_obs = []
    manual_rewards = []

    # -------- RL trajectory --------
    obs, _ = _reset_env(env)
    for t in range(num_steps):
        obs_flat = np.array(obs, dtype=np.float32).reshape(-1)
        rl_action, _ = model.predict(obs_flat, deterministic=True)

        obs, reward, done, _ = _step_env(env, rl_action)
        rl_next_obs.append(obs)
        rl_rewards.append(reward)
        if done:
            break

    # -------- Manual trajectory --------
    manual_env = LiquidityEnv()
    obs_m, _ = _reset_env(manual_env)
    for t in range(num_steps):
        action = manual_actions[min(t, len(manual_actions) - 1)]
        obs_m, reward_m, done_m, _ = _step_env(manual_env, action)
        manual_next_obs.append(obs_m)
        manual_rewards.append(reward_m)
        if done_m:
            break

    steps = min(len(rl_next_obs), len(manual_next_obs))
    rl_next_obs = rl_next_obs[:steps]
    rl_rewards = rl_rewards[:steps]
    manual_next_obs = manual_next_obs[:steps]
    manual_rewards = manual_rewards[:steps]

    df = pd.DataFrame({
        "step": range(steps),
        "rl_liquidity": [o[0] for o in rl_next_obs],
        "rl_volatility": [o[1] for o in rl_next_obs],
        "rl_apy": [o[2] for o in rl_next_obs],
        "rl_reward": rl_rewards,
        "manual_liquidity": [o[0] for o in manual_next_obs],
        "manual_volatility": [o[1] for o in manual_next_obs],
        "manual_apy": [o[2] for o in manual_next_obs],
        "manual_reward": manual_rewards,
    })
    return df
