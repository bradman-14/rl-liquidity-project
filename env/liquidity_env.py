from typing import Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class LiquidityEnv(gym.Env):
    """
    Simple liquidity-pool environment.

    State (observation):
        [liquidity, volatility, current_apy]
        - liquidity: 0.0 to 1.0  (fraction of "max" liquidity)
        - volatility: 0.0 to 1.0 (normalized)
        - current_apy: 0.0 to 0.5 (0% to 50% annual yield)

    Actions:
        Discrete {0,1,2,3,4} mapped to APY changes:
            0 -> -0.002 (-20 bp)
            1 -> -0.001 (-10 bp)
            2 ->  0.0   ( 0 bp)
            3 -> +0.001 (+10 bp)
            4 -> +0.002 (+20 bp)

    Reward:
        R = A * liquidity - B * volatility - C * current_apy
        Encourages:
        - higher liquidity,
        - lower volatility,
        - lower cost of rewards (APY).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Observation space: 3 continuous values
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 0.5], dtype=np.float32),  # APY up to 50%
        )

        # Discrete action space with 5 actions
        self.action_space = spaces.Discrete(5)

        # APY limits (you can tune these later)
        self.min_apy = 0.02  # 2%
        self.max_apy = 0.25  # 25%

        # Reward weights
        self.A = 1.0  # weight for liquidity
        self.B = 0.5  # weight for volatility
        self.C = 0.2  # weight for APY cost

        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        # Standard Gymnasium reset pattern
        super().reset(seed=seed)

        # Start from moderate conditions
        self.liquidity = 0.5
        self.volatility = 0.2
        self.current_apy = 0.05  # 5%

        self.step_count = 0

        obs = np.array(
            [self.liquidity, self.volatility, self.current_apy],
            dtype=np.float32,
        )
        return obs, {}

    def step(self, action: int):
        # Map action index to delta APY
        delta_map = {
            0: -0.002,  # -20 bp
            1: -0.001,  # -10 bp
            2: 0.0,     #  0 bp
            3: 0.001,   # +10 bp
            4: 0.002,   # +20 bp
        }
        delta_apy = delta_map[int(action)]

        # Update APY with min/max bounds
        self.current_apy = float(
            np.clip(self.current_apy + delta_apy, self.min_apy, self.max_apy)
        )

        # -----------------------------
        # Simple toy market dynamics
        # -----------------------------
        # Higher APY -> more liquidity
        # anchor_apy is a "neutral" level
        anchor_apy = 0.05
        liquidity_change = 0.5 * (self.current_apy - anchor_apy)
        self.liquidity = float(
            np.clip(self.liquidity + liquidity_change, 0.0, 1.0)
        )

        # Volatility: decreases when liquidity is high, plus noise
        vol_noise = np.random.normal(0.0, 0.01)
        self.volatility = float(
            np.clip(self.volatility - 0.1 * self.liquidity + vol_noise, 0.0, 1.0)
        )

        # Reward combines all three components
        reward = (
            self.A * self.liquidity
            - self.B * self.volatility
            - self.C * self.current_apy
        )

        self.step_count += 1
        terminated = self.step_count >= 500  # episode ends after 500 steps
        truncated = False

        obs = np.array(
            [self.liquidity, self.volatility, self.current_apy],
            dtype=np.float32,
        )
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        # Simple print for debugging
        print(
            f"Step={self.step_count} | "
            f"Liquidity={self.liquidity:.3f}, "
            f"Volatility={self.volatility:.3f}, "
            f"APY={self.current_apy:.4f}"
        )
