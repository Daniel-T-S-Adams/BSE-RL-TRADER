import numpy as np 
import random
from BSE import Trader, Order
from typing import List, Dict, DefaultDict
import gymnasium as gym
from gymnasium import Space
from gymnasium import spaces
from collections import defaultdict
from tqdm import tqdm


class RLAgent(Trader):
    def __init__(self, ttype, tid, balance, params, time, action_space: spaces.Space, 
                 obs_space: spaces.Space, gamma=1.0, epsilon=0.1):
        
        super().__init__(ttype, tid, balance, params, time)
        self.action_space = action_space
        self.obs_space = obs_space
        self.gamma: float = gamma
        self.epsilon: float = epsilon

        self.num_actions = spaces.flatdim(action_space)
        self.q_table: DefaultDict = defaultdict(lambda: 0)
        self.sa_counts = {}
        self.current_obs = None

    
    def set_obs(self, obs):
        self.current_obs = obs

        
    # implement epsilon-greedy action selection
    def act(self, obs: int) -> int:
        obs = tuple(obs)
        if random.uniform(0, 1) < self.epsilon:
            # Explore - sample a random action
            return self.action_space.sample()
        else:
            # Exploit - choose the action with the highest probability
            return max(list(range(self.action_space.n)), key = lambda x: self.q_table[(obs, x)])


    def learn(self, obs: List[int], actions: List[int], rewards: List[float]) -> Dict:
        
        traj_length = len(rewards)
        G = 0
        state_action_list = list(zip(obs, actions))
        updated_values = {}
        
        # Iterate over the trajectory backwards
        for t in range(traj_length - 1, -1, -1):
            state_action_pair = (tuple(obs[t]), actions[t])

            # Check if this is the first visit to the state-action pair
            if state_action_pair not in state_action_list[:t]:
                G = self.gamma*G + rewards[t]

                # Monte-Carlo update rule
                self.sa_counts[state_action_pair] = self.sa_counts.get(state_action_pair, 0) + 1
                self.q_table[state_action_pair] += (
                    G - self.q_table[state_action_pair]
                    ) / self.sa_counts.get(state_action_pair, 0)
                
                updated_values[state_action_pair] = self.q_table[state_action_pair]
      
        return updated_values
    

    # We need to be allowed to give obs as an input parameter so that
    # the agent can pick the best action given the current state
    def getorder(self, time, countdown, lob):        
        if len(self.orders) < 1:
            order = None
        else:
            order_type = self.orders[0].otype
            # return the best action following a greedy policy
            obs = tuple(self.current_obs)
            quote = max(list(range(self.action_space.n)), key = lambda x: self.q_table[(obs, x)])

        order = Order(self.tid, order_type, quote, self.orders[0].qty, time, lob['QID'])

        return order


