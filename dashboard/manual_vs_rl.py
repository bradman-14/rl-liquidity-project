import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env import LiquidityEnv

@st.cache_resource
def load_model():
    return PPO.load("rl/models/ppo_liquidity.zip")

def run_comparison(manual_actions, num_steps=500):
    env = LiquidityEnv()
    model = load_model()
    
    rl_obs, rl_actions, rl_next_obs, rl_rewards = [], [], [], []
    manual_obs, manual_next_obs, manual_rewards = [], [], []
    
    obs = env.reset()
    
    for t in range(num_steps):
        # RL trajectory
        rl_action, _ = model.predict(obs, deterministic=True)
        next_obs_rl, reward_rl, _, _ = env.step(rl_action)
        rl_obs.append(obs)
        rl_actions.append(rl_action)
        rl_next_obs.append(next_obs_rl)
        rl_rewards.append(reward_rl)
        obs = next_obs_rl
        
        # Manual trajectory (independent env)
        manual_env = LiquidityEnv()
        manual_obs_t = manual_env.reset()
        manual_action = manual_actions[min(t, len(manual_actions)-1)]
        next_obs_manual, reward_manual, _, _ = manual_env.step(manual_action)
        manual_obs.append(manual_obs_t)
        manual_next_obs.append(next_obs_manual)
        manual_rewards.append(reward_manual)
    
    # Create comparison DataFrame
    df = pd.DataFrame({
        'step': range(num_steps),
        'rl_liquidity': [o[0] for o in rl_next_obs],
        'rl_volatility': [o[1] for o in rl_next_obs],
        'rl_apy': [o[2] for o in rl_next_obs],
        'rl_reward': rl_rewards,
        'manual_liquidity': [o[0] for o in manual_next_obs],
        'manual_volatility': [o[1] for o in manual_next_obs],
        'manual_apy': [o[2] for o in manual_next_obs],
        'manual_reward': manual_rewards,
    })
    return df
