import BSE
from BSE import Exchange, market_session
from typing import Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv

class Environment(gym.Env):
    def __init__(self):
        self.min_price = BSE.bse_sys_minprice
        self.max_price = BSE.bse_sys_maxprice
        self.price_range = self.max_price - self.min_price
        self.num_traders = 20
        self.max_length = self.num_traders
        self.trader_type = None
        self.order = 0

        # Create low and high bounds with padding for variable length lists
        low = np.array([[self.min_price, 1]] * self.max_length + [[0, 0]] * (self.max_length - 1))
        high = np.array([[self.max_price, self.num_traders]] * self.max_length + [[0, 0]] * (self.max_length - 1))

        self.observation_space = spaces.Dict(
            {'Bids': spaces.Box(
                    low=low,
                    high=high,
                    dtype=np.float32),
            'Asks': spaces.Box(
                low=low,
                high=high,
                dtype=np.float32)
            })

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
    def get_order(self, order_schedule):
        self.order = np.random.randint(order_schedule[0], order_schedule[1])


    def csv_to_dict(self, file_path):
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

        self.lob = data_dict
        return self.lob


    def analyse_lob(self, file_path):
        """
        Analyse the limit order book from a CSV file.
        
        Parameters:
        file_path (str): Path to the CSV file.
        
        Returns:
        dict: Analysis of the limit order book.
        """
        self.lob = self.csv_to_dict(file_path)
        analysis = {}

        for time, data in self.lob.items():
            bids = [bid[0] for bid in data['Bids']]
            asks = [ask[0] for ask in data['Asks']]

            best_bid = max(bids) if bids else np.nan
            worst_bid = min(bids) if bids else np.nan
            avg_bid = np.mean(bids) if bids else np.nan
            var_bid = np.var(bids) if bids else np.nan

            best_ask = min(asks) if asks else np.nan
            worst_ask = max(asks) if asks else np.nan
            avg_ask = np.mean(asks) if asks else np.nan
            var_ask = np.var(asks) if asks else np.nan

            analysis[time] = [best_bid, best_ask, worst_bid, worst_ask, avg_bid, avg_ask, var_bid, var_ask]

        self.lob_summary = analysis
        # return analysis

    
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
        # i don't think this function is needed
        pass


file_path = 'bse_d010_i15_0001_LOB_frames.csv'
raya = Environment()
raya.initialise_trader()
# raya.get_order([100, 200])
raya.order = 40
# raya.csv_to_dict('data.csv')
# print(raya.lob)
raya.analyse_lob(file_path)
lob_summary = raya.lob_summary
print(lob_summary[6.275])
# quote = 54.0
# reward = raya.calculate_reward(quote)
# print(f"raya is a {raya.trader_type}. the order is {raya.order}. the reward is {reward}")



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
