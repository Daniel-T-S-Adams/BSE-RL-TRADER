import numpy as np 
import random
import environment
from environment import AuctionEnv, lob
import BSE
from BSE import Trader, Order
from typing import List, Dict, DefaultDict
import gymnasium as gym
from gymnasium import Space
from gymnasium import spaces
from collections import defaultdict
from tqdm import tqdm


class RLAgent(Trader):
    def __init__(self, action_space: Space, obs_space: Space,
                 gamma: 0.0, epsilon: 0.1):
        # self.ttype = ttype          # what type / strategy this trader is
        # self.tid = tid              # trader unique ID code
        # self.balance = balance      # money in the bank
        # self.params = params        # parameters/extras associated with this trader-type or individual trader.
        # self.blotter = []           # record of trades executed
        # self.blotter_length = 100   # maximum length of blotter
        # self.orders = []            # customer orders currently being worked (fixed at len=1 in BSE1.x)
        # self.n_quotes = 0           # number of quotes live on LOB
        # self.birthtime = time       # used when calculating age of a trader/strategy
        # self.profitpertime = 0      # profit per unit time
        # self.profit_mintime = 60    # minimum duration in seconds for calculating profitpertime
        # self.n_trades = 0           # how many trades has this trader done?
        # self.lastquote = None       # record of what its last quote was
        
        self.action_space = action_space
        self.obs_space = obs_space
        self.num_actions = spaces.flatdim(action_space)
        
        self.gamma: float = gamma
        self.epsilon: float = epsilon

        self.q_table: DefaultDict = defaultdict(lambda: 0)
        self.sa_counts = {}
        
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
            quote = max(list(range(self.action_space.n)), key = lambda x: self.q_table[(obs, x)])

        order = Order(self.tid, order_type, quote, self.orders[0].qty, time, lob['QID'])

        return order


