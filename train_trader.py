import random
import csv
import numpy as np 
from tqdm import tqdm
from BSE import market_session
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from q_table_data import load_q_table, dump_q_table, update_q_table
from epsilon_scheduling import epsilon_decay

# Plotting
from matplotlib import pyplot as plt
from Plotting import plot_avg_profit

#file handling
from typing import List, Tuple
import shutil
import csv
import os

# Importing Global Parameters
from GlobalParameters import CONFIG


###### The functions

def train(total_eps: int, market_params: tuple, test_freq: int, epsilon_start: float) -> DefaultDict:
    saved_stats = [] # start a list which we will be appending. 
    # Start the GPI iterations at 1 
    GPI_iter = 1
    print(f"Starting GPI iteration {GPI_iter}")
    
    epsilon = epsilon_start
    
    # Initialize the counts and returns dictionaries
    sa_counts = defaultdict(lambda: 0)
    sa_returns = defaultdict(lambda: 0)
    
    for episode in range(1, total_eps + 1):
        # Run the market session once, it saves a CSV file
        market_session(*market_params)
        
        # Check if there's a sell trader, and then read the episode infomation from its CSV file
        # note this means every episode we need to write and read a CSV
        try:
            file = 'episode_seller.csv'
            obs_list, action_list, reward_list = load_episode_data(file)
        except Exception as e:
            print(f"Error loading seller episode {episode}: {e}")
            pass
        
        # Update the count and returns
        try:
            sa_counts, sa_returns = learn(obs_list, action_list, reward_list, sa_counts, sa_returns)
        except Exception as e:
            print(f"Error computing new count and returns for seller episode {episode}: {e}")
            pass

        # If we have done the designated number of episodes for this policy evaluation, compute the q_values i.e. q_table
        if episode % CONFIG["eps_per_evaluation"] == 0: 
            
            # Compute the q_table by averaging and save it to a CSV file name q_table_seller.csv
            save = True
            average(sa_counts, sa_returns, save)
            
            # Save the sellers q_table file for this GPI_iter (to keep track)
            new_file_name = os.path.join(CONFIG["q_tables"],f'q_table_seller_after_GPI_{GPI_iter}.csv')
            shutil.copy('q_table_seller.csv', new_file_name)  
            
            # Save sa_counts to a CSV file for this GPI_iter (to keep track)
            sa_counts_filename = os.path.join(CONFIG["counts"],f'sa_counts_after_GPI_{GPI_iter}.csv')
            save_sa_counts_to_csv(sa_counts, sa_counts_filename)
            
            # Restart the counts and returns for the next iteration of policy evalutation
            sa_counts = defaultdict(lambda: 0)
            sa_returns = defaultdict(lambda: 0)
            
            # Update epsilon for the next iteration of policy evaluation
            old_epsilon = epsilon   # save incase we want to test this policy  
            epsilon = epsilon_decay('linear', GPI_iter, CONFIG["num_GPI_iter"], epsilon_start, 0.05)
            market_params[3]['sellers'][1][2]['epsilon'] = epsilon
            print(f"New epsilon: {epsilon}")
            
            
            GPI_iter += 1
            print(f"Starting GPI iteration {GPI_iter}")
        
        # Perform a test of these policies performance every `test_freq` episodes
        if episode % test_freq == 0:
            print(f"Testing the Performance after GPI iteration {GPI_iter}")
            
            
            cumulative_stats = test_policy(
            episodes=CONFIG['test_episodes'], market_params=market_params, 
            q_table='q_table_seller.csv', file='episode_seller.csv', epsilon = old_epsilon)
            
            
            for ttype in cumulative_stats:
                print(f"Performance Test: GPI Iter {GPI_iter}, {ttype} average profit: {cumulative_stats[ttype]['avg_profit']}")
                
            saved_stats.append(cumulative_stats)
            
    return saved_stats # saved stats is a list of dictionaries one for each GPI iteration. 

   


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


# This takes a file name for a CSV file, containing the episode data, state,action,reward. It reads this file converts to lists
def load_episode_data(file: str) -> Tuple[List, List, List]:
    obs_list, action_list, reward_list = [], [], []

    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            obs_str = row[0].strip('()').split(", ")[1:]
            obs_list.append(np.array([float(x.strip("'")) for x in obs_str]))        # Convert the string values to floats
            action_list.append((float(row[1])))
            reward_list.append(float(row[2]))

    return obs_list, action_list, reward_list



def save_dict_to_csv(filename: str, data: dict):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['State', 'Action', 'Q-Value'])
        # Write the data
        for (state, action), q_value in data.items():
            writer.writerow([state, action, q_value])

# Function to save sa_counts to a CSV file
def save_sa_counts_to_csv(sa_counts, filename):
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header
        writer.writerow(['State', 'Action', 'Count'])
        # Write the data
        for (state, action), count in sa_counts.items():
            writer.writerow([state, action, count])


def learn(obs: List[int], actions: List[int], rewards: List[float], sa_counts, sa_returns) -> Tuple[DefaultDict, DefaultDict]:
    traj_length = len(rewards)   
    
    if traj_length == 0:
        return sa_counts, sa_returns

    # Precompute returns G for every timestep
    G = [ 0 for n in range(traj_length) ]
    G[-1] = rewards[-1]
    for t in range(traj_length - 2, -1, -1):
        G[t] = rewards[t] + CONFIG["gamma"] * G[t + 1] 
    
    # Update returns and counts 
    for t in range(traj_length):
        state_action_pair = (tuple(obs[t]), actions[t])
        sa_counts[state_action_pair] += 1
        sa_returns[state_action_pair] += G[t]
        
    
    return sa_counts, sa_returns


def average(sa_counts, sa_returns, save=False):
    # Ensure all keys in sa_returns are in sa_counts to avoid KeyError
    for key in sa_returns:
        if key not in sa_counts:
            raise KeyError(f"Key {key} found in sa_returns but not in sa_counts.")
        if sa_counts[key] == 0:
            raise ValueError(f"Count for key {key} is zero, cannot divide by zero.")

    # Create a new dictionary with the results of the division
    sa_average = {key: sa_returns[key] / sa_counts[key] for key in sa_returns}
    
    # Sort the dictionary by state
    sorted_sa_average = sorted(sa_average.items(), key=lambda x: x[0][0])
    
    # count the percentage of returns that are zero, and then print that number
    total_entries = len(sa_average)
    zero_entries = sum(1 for value in sa_average.values() if value == 0)
    zero_percentage = (zero_entries / total_entries) * 100
    print(f"Percentage of zero values: {zero_percentage:.2f}%")
    
    # Save it as a CSV file if set to True
    if save:
        with open('q_table_seller.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['State', 'Action', 'Q-Value'])
            # Write the data
            for (state, action), q_value in sorted_sa_average:
                writer.writerow([state, action, q_value])
        
    return sa_average

def test_policy(episodes: int, market_params: tuple, q_table: DefaultDict, file, epsilon) -> dict:

    updated_market_params = list(market_params)    
    # if file == 'q_table_buyer.csv':
    #     updated_market_params[3]['buyers'][0][2]['q_table_buyer'] = 'q_table_buyer.csv'
    #     updated_market_params[3]['buyers'][0][2]['epsilon'] = 0.0                           # No exploring
    # elif file == 'q_table_seller.csv':
    updated_market_params[3]['sellers'][1][2]['q_table_seller'] = 'q_table_seller.csv'
    updated_market_params[3]['sellers'][1][2]['epsilon'] = epsilon                          # No exploring

    # initialize an empty dictionary to store cumulative average profit
    cumulative_stats = {}
    # for storing previous profit
    previous_avg_profit = None
    for episode in range(episodes):
        # Run the market session
        market_session(*updated_market_params)
        # Read the average profit file at the final timestep of each market session
        current_stats = read_average_profit('session_1_avg_balance.csv')
        # getting a cumulative tally of the average profit for each trader type 
        update_cumulative_average_profit(cumulative_stats, current_stats)

        # Calculate average profit for the current episode
        current_avg_profit = sum([cumulative_stats[ttype]['avg_profit'] for ttype in cumulative_stats])/(episode+1)
        # Check for convergence every 100 steps
        if episode % 100 == 0:
            if previous_avg_profit is not None:
                profit_change = abs(current_avg_profit - previous_avg_profit)
                if profit_change <= 0.00005:
                    print(f"Convergence achieved at episode {episode} with profit change {profit_change}")
                    # get the average over all episodes
                    for ttype in cumulative_stats:
                        cumulative_stats[ttype]['avg_profit'] /= (episode+1)
                    return cumulative_stats
            
            if episode % 100 == 0 or episode == 0:
                previous_avg_profit = current_avg_profit
       
    # get the average over all episodes if we dont converge
    print(f"Did not converge after {episodes} episodes")
    for ttype in cumulative_stats:
        cumulative_stats[ttype]['avg_profit'] /= episodes
        
       
    return cumulative_stats




