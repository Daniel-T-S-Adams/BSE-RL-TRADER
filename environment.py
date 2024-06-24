import BSE
from BSE import Exchange, market_session
import random
from typing import Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv

class AuctionEnv(gym.Env):
    def __init__(self, lob, price_range=[100, 200], n_values=10):
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
        self.lob = lob

    # Randomly activate trader as a buyer or seller
    def initialise_trader(self):
        if np.random.randint(2) == 0:
            self.trader_type = 'Buyer'
        else:
            self.trader_type = 'Buyer'

    # Generates a customer order that is a random
    # integer from the given range
    # should the orders be normally distributed?
    def get_order(self):
        self.order = np.random.randint(self.min_price, self.max_price+1)


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
                    if quote > price:
                        reward = self.order - price
                        break  # Exit loop once a match is found
            else:
                # Go through the bids in the lob and see if it matches with the quote
                for bid in data['Bids']:
                    price = bid[0]
                    if quote < price:
                        reward = quote - self.order
                        break  # Exit loop once a match is found

        return reward

    
    def step(self, action):
        
        observation = np.zeros(self.observation_space.shape)
        reward = self.calculate_reward(action)
        # reward = 1.0

        terminated = True
        truncated = True
        info = {}

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.initialise_trader()
        self.get_order()

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


def csv_to_dict(file_path):
        """
        Convert an inhomogeneous CSV file to a dictionary.
        
        Parameters:
        file_path (str): Path to the CSV file.
        
        Returns:
        dict: Dictionary representation of the CSV data.
        """
        data_dict = {}

        # Step 1: Read the CSV file
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            first_row = True
            for row in csv_reader:
                if first_row:
                    first_row = False
                    continue  # Skip the first row

                # Step 2: Extract time
                try:
                    time = float(row[0])
                except ValueError:
                    continue  # Skip rows with invalid time values

                # Initialize the dictionary for the current time
                if time not in data_dict:
                    data_dict[time] = {'Bids': [], 'Asks': []}

                # Step 3: Process the row to extract bids and asks
                i = 1  # Start after the time column
                while i < len(row):
                    if row[i] == ' Bid:':
                        bid_count = int(row[i + 1])  # The count of bids
                        i += 2  # Skip the 'Bid:' and the count
                        for _ in range(bid_count):
                            try:
                                price = float(row[i])
                                quantity = int(row[i + 1])
                                data_dict[time]['Bids'].append([price, quantity])
                            except (ValueError, IndexError):
                                pass  # Skip invalid entries
                            i += 2

                    elif row[i] == 'Ask:':
                        ask_count = int(row[i + 1])  # The count of asks
                        i += 2  # Skip the 'Ask:' and the count
                        for _ in range(ask_count):
                            try:
                                price = float(row[i])
                                quantity = int(row[i + 1])
                                data_dict[time]['Asks'].append([price, quantity])
                            except (ValueError, IndexError):
                                pass  # Skip invalid entries
                            i += 2

                    else:
                        i += 1

        return data_dict


file_path = 'data.csv'
lob = csv_to_dict(file_path)
raya = AuctionEnv(lob, price_range=[0,500])

# obs = raya.reset()
# print(obs)
# print(f"Raya is a {raya.trader_type} and the customer order is {raya.order}")

# for _ in range(1):
#     action = raya.action_space.sample()
#     print(action)
#     print(raya.step(action)) # take a random action