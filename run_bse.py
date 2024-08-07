# A FILE FOR RUNNING THE BSE WHEN EVERYTHING IS TRAINED ETC.

import random
import csv
import numpy as np 
from tqdm import tqdm
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from q_table_data import load_q_table, dump_q_table
from epsilon_scheduling import epsilon_decay
import shutil

def read_average_profit(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Get the final line
    final_line = lines[-1]

    # Split the final line into components
    components = final_line.strip().split(', ')

    # Initialize an empty dictionary to store trader stats
    trader_stats = {}
    # Skip the first four components (sesid, time, best bid, best offer)
    index = 4
    # Iterate over the components to extract trader stats
    while index < len(components):
        ttype = components[index]
        total_profit = float(components[index + 1])
        num_traders = int(components[index + 2])
        # Clean up the string before converting to float by removing commas
        avg_profit = float(components[index + 3].replace(',', ''))
        trader_stats[ttype] = {
            'total_profit': total_profit,
            'num_traders': num_traders,
            'avg_profit': avg_profit
        }
        index += 4

    return trader_stats

def update_cumulative_average_profit(cumulative_stats, new_stats):
    for ttype, stats in new_stats.items():
        if ttype in cumulative_stats:
            cumulative_stats[ttype]['avg_profit'] += stats['avg_profit']
        else:
            cumulative_stats[ttype] = {
                'total_profit': stats['total_profit'],
                'num_traders': stats['num_traders'],
                'avg_profit': stats['avg_profit']
            }

def run_market(total_eps: int, market_params: tuple) -> DefaultDict:
    # initialize an empty dictionary to store cumulative average profit
    cumulative_stats = {}
    for episode in range(1, total_eps + 1):
        # Run the market session
        market_session(*market_params)
        # Read the average profit file at the final timestep of each market session
        current_stats = read_average_profit('session_1_avg_balance.csv')
        # getting a cumulative tally of the average profit for each trader type 
        update_cumulative_average_profit(cumulative_stats, current_stats)
    
    # get the average over all episodes
    for ttype in cumulative_stats:
        cumulative_stats[ttype]['avg_profit'] /= total_eps
        
    # print the values of the cumulative average profit
    for ttype in cumulative_stats:
               print(f"{ttype} average profit: {cumulative_stats[ttype]['avg_profit']}")
    
    
    return cumulative_stats




# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 30.0

sellers_spec = [('GVWY',19), ('RL', 1, {'epsilon': 0.0})]
buyers_spec = [('GVWY',20)]


trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

range1 = (1, 4)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (1, 4)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 30

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-fixed'}

dump_flags = {'dump_strats': True, 'dump_lobs': True, 'dump_avgbals': True, 'dump_tape': True, 'dump_blotters': True}
verbose = False


# Training the RL agent with evaluation
q_table = run_market(total_eps=5000, 
                market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose))