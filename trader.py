import numpy as np 
import environment
from environment import AuctionEnv
from typing import Dict, DefaultDict
import gymnasium as gym
from gymnasium import Space
from gymnasium import spaces
from collections import defaultdict

class Trader():
    def __init__(self, action_space: Space, obs_space: Space,
                 gamma: float, epsilon: float):
        self.action_space = action_space
        self.obs_space = obs_space
        self.num_actions = spaces.flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        # self.q_table: DefaultDict = defaultdict(lambda: 0)

    # implement epsilon-greedy action selection
    def act(self, obs: int) -> int:

        pass
        

min_price = 100
max_price = 200
