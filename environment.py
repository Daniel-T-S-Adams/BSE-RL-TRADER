import BSE
from BSE import Exchange, market_session
import random
from typing import Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv

class AuctionEnv(gym.Env):
    def __init__(self, price_range=[100, 200], n_values=10):
        super(AuctionEnv, self).__init__()
        self.min_price = price_range[0]
        self.max_price = price_range[1]
        self.price_range = price_range[1] - price_range[0]
        self.trader_type = None
        self.order = 0
        self.n_values = n_values

        self.observation_space = spaces.MultiDiscrete([n_values, n_values, n_values, n_values])
        
        self.action_space = spaces.Discrete(self.price_range, start=self.min_price)

        self.state = None

    # Randomly activate trader as a buyer or seller
    def initialise_trader(self):
        if np.random.randint(2) == 0:
            self.trader_type = 'Buyer'
        else:
            self.trader_type = 'Seller'

    # Generates a customer order that is a random
    # integer from the given range
    # should the orders be normally distributed?
    def get_order(self):
        self.order = np.random.randint(self.price_range[0], self.price_range[1])


    def calculate_reward(self, quote):
        """
        Calculate the reward based on the limit order book (lob), order, and quote.

        Parameters:
        lob (dict): The limit order book dictionary.
        order (float): The order value.
        quote (float): The quote value.

        Returns:
        float: The calculated reward.
        """
        reward = 0

        for time, data in self.lob.items():
            if self.trader_type == 'Buyer':
                # Go through the asks in the lob and see if it matches with the quote
                for ask in data['Asks']:
                    price = ask[0]
                    if price == quote:
                        reward = self.order - quote
                        break  # Exit loop once a match is found
            else:
                # Go through the bids in the lob and see if it matches with the quote
                for bid in data['Bids']:
                    price = bid[0]
                    if price == quote:
                        reward = quote - self.order
                        break  # Exit loop once a match is found

        return reward

    
    def step(self, action):
        
        observation = np.zeros(self.observation_space.shape)
        # reward = self.calculate_reward(action)
        reward = 1.0

        terminated = True
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        step = int(self.price_range / self.n_values)
        
        best_bid = random.randrange(self.min_price, self.max_price, step)
        best_ask = random.randrange(self.min_price+step, self.max_price+step, step)

        # these two need to be changed so that the 
        # worst_bid < best_bid and worst_ask > best_ask
        worst_bid = random.randrange(self.min_price, self.max_price, step)
        worst_ask = random.randrange(self.min_price+step, self.max_price+step, step)

        obs = np.array([best_bid, best_ask, worst_bid, worst_ask])
        info = {}

        return obs, info


# file_path = 'bse_d010_i15_0001_LOB_frames.csv'
# raya = AuctionEnv()
# raya.initialise_trader()

# obs = raya.reset()
# print(obs)

# for _ in range(10):
#     action = raya.action_space.sample()
#     # print(action)
#     print(raya.step(action)) # take a random action
