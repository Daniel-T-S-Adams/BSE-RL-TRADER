import BSE
from BSE import Exchange, market_session
import random
from typing import Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv

class Environment(gym.Env):
    def __init__(self, price_range=[100, 200], n_values=10):
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

        return observation, reward, terminated


    def reset(self):

        best_bid = random.randrange(105, 200, 10)
        best_ask = random.randrange(105, 200, 10)
        worst_bid = random.randrange(100, 200, 10)
        worst_ask = random.randrange(110, 200, 10)

        return [best_bid, best_ask, worst_bid, worst_ask]


# file_path = 'bse_d010_i15_0001_LOB_frames.csv'
raya = Environment()
raya.initialise_trader()

obs = raya.reset()
print(obs)
for _ in range(10):
    action = raya.action_space.sample()
    # print(action)
    print(raya.step(action)) # take a random action



# if __name__ == "__main__":
#     n_days = 10
#     start_time = 0.0
#     end_time = 60.0
#     duration = end_time - start_time

#     range1 = (50, 150)
#     supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

#     range2 = (50, 150)
#     demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

#     # new customer orders arrive at each trader approx once every order_interval seconds
#     order_interval = 15

#     order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
#                     'interval': order_interval, 'timemode': 'drip-poisson'}

#     verbose = False

#     # n_trials is how many trials (i.e. market sessions) to run in total
#     n_trials = 1

#     # n_recorded is how many trials (i.e. market sessions) to write full data-files for
#     n_trials_recorded = 5

#     trial = 1

#     while trial < (n_trials+1):

#         # create unique i.d. string for this trial
#         trial_id = 'bse_d%03d_i%02d_%04d' % (n_days, order_interval, trial)

#         buyers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]
#         sellers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]

#         traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

#         if trial > n_trials_recorded:
#             dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
#                             'dump_avgbals': False, 'dump_tape': False}
#         else:
#             dump_flags = {'dump_blotters': True, 'dump_lobs': False, 'dump_strats': True,
#                             'dump_avgbals': True, 'dump_tape': True}

#         market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

#         trial = trial + 1
