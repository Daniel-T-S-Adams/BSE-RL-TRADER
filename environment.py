import BSE
from BSE import Exchange, Trader, market_session
from trader import RLAgent
import random
from typing import Dict
import gymnasium as gym
from gymnasium import Space
from gymnasium import spaces
import numpy as np
import csv

class AuctionEnv(gym.Env, Trader):
    def __init__(self, action_space: Space, obs_space: Space, gamma: 0.0, epsilon: 0.1, bins=5):
        # Initialise both parent classes
        gym.Env.__init__(self)
        Trader.__init__(self, action_space, obs_space, gamma, epsilon)

        self.bins = bins
        # self.price_range = price_range[1] - price_range[0]

        self.trader_type = 'Buyer'
        self.start_time = BSE.start_time
        self.end_time = BSE.end_time
        self.time_interval = self.end_time - self.start_time
        self.order_range = BSE.demand_schedule['ranges'][1] - BSE.demand_schedule['ranges'][0]
        self.min_price = self.order_range[0]
        self.max_price = self.order_range[1]
        self.time = 0
        self.order = 0
        self.best_bid = 0
        self.best_ask = 0
        self.worst_bid = 0
        self.worst_ask = 0
        self.state = None

        self.observation_space = spaces.MultiDiscrete([int(self.time_interval), self.order_range, bins, bins, bins, bins])
        self.action_space = spaces.Discrete(self.price_range, start=self.min_price)


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

    
    def _get_obs(self):
        return np.array([self.time, self.order, self.best_bid, self.best_ask, self.worst_bid, self.worst_ask])


    def set_additional_params(self, lob):
        self.lob = lob


    def calc_average_price(self, list):
        # Calculate the total price contribution and total quantity
        total_price = sum(price * quantity for price, quantity in list)
        total_quantity = sum(quantity for price, quantity in list)
        
        # Calculate the weighted average price
        weighted_average_price = total_price / total_quantity if total_quantity else 0
        
        return weighted_average_price   


    def bin_average(self, value):
        # Calculate bin width
        bin_width = (self.max_price - self.min_price) / self.bins

        # Determine the bin index for the value
        bin_index = int((value - self.min_price) / bin_width)

        # Ensure the value is placed in the last bin if it falls on max_price
        if bin_index == self.bins:
            bin_index -= 1

        # Calculate the average of the bin range
        bin_start = self.min_price + bin_index * bin_width
        bin_end = bin_start + bin_width
        bin_average = (bin_start + bin_end) / 2

        return int(bin_average)


    def step(self, action):
        last_trade_time = self.blotter[-1]['time']
        self.set_additional_params(lob)

        # We have a new customer order
        if self.orders[0] != self.order:
            terminated = True
            # Check if there was a trade
            if last_trade_time == self.time:
                transaction_price = self.blotter[-1]['price']
                reward = self.orders[0].price - transaction_price

            self.time += 1
            self.order = self.orders[0]                                                        # Update to new customer order
            self.best_bid = self.bin_average(lob['bids']['best'])                              # Update best bid
            self.best_ask = self.bin_average(lob['asks']['best'])                              # Update best ask
            self.worst_bid = self.bin_average(lob['bids']['worst'])                            # Update worst bid
            self.worst_ask = self.bin_average(lob['asks']['worst'])                            # Update worst ask
            # self.avg_bid = self.calc_average_price(lob['bids']['lob'])                       # Calculate new average big
            # self.avg_ask = self.calc_average_price(lob['asks']['lob'])                       # Calculate new average ask
            observation = np.array([self.time, self.order, self.best_bid, 
                                    self.best_ask, self.worst_bid, self.worst_ask])

        else:
            # Don't update bid and ask metrics
            self.time += 1
            terminated = False
            reward = 0.0
            observation = np.array([self.time, self.order, self.best_bid, 
                                    self.best_ask, self.worst_bid, self.worst_ask])
        
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        step = int(self.price_range / self.n_values)
        
        best_bid = random.randrange(self.min_price, self.max_price, step)
        best_ask = random.randrange(self.min_price+step, self.max_price+step, step)

        # Randomly sets worst bid and ask such that
        # worst_bid < best_bid and worst_ask > best_ask
        worst_bid = random.randrange(self.min_price, best_bid+step, step)
        worst_ask = random.randrange(best_ask, self.max_price+step, step)

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