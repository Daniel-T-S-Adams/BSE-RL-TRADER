import random
import csv
import numpy as np 
from tqdm import tqdm
from BSE import market_session
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from q_table_data import load_q_table, dump_q_table, update_q_table
from epsilon_scheduling import epsilon_decay
import ast


#file handling
from typing import List, Tuple
import shutil
import csv
import os

# Importing Global Parameters
from GlobalParameters import CONFIG


###### The functions

def train(total_eps: int, market_params: tuple, test_freq: int, epsilon_start: float) -> DefaultDict:
     
     
    ## Initialize everything: ##
    
    # Start the GPI iterations at 1 
    GPI_iter = 1
    print(f"Starting GPI iteration {GPI_iter}")
    
    epsilon = epsilon_start
    
    # Initialize the counts and returns dictionaries
    sa_counts = defaultdict(lambda: 0)
    sa_returns = defaultdict(lambda: 0)
    
    # empty dictionary 
    Q_old = {}
    
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
            
            
            # Compute the Q_new by averaging 
            Q_new = average(sa_counts, sa_returns)
            # Compute the next q_table by an incremental update
            next_q_table = incremental_update(Q_new, Q_old, CONFIG["alpha"])    
            # save this Q_table for the next increment step
            Q_old = next_q_table
            # Save the new q_table dictionary to a CSV file named q_table_seller.csv
            save_dict_to_csv(next_q_table)
            
            # Save the sellers q_table file for this GPI_iter (to keep track)
            new_file_name = os.path.join(CONFIG["q_tables"], f'q_table_seller_after_GPI_{GPI_iter}.csv')
            shutil.copy('q_table_seller.csv', new_file_name)  
            
            # Save sa_counts to a CSV file for this GPI_iter (to keep track)
            sa_counts_filename = os.path.join(CONFIG["counts"], f'sa_counts_after_GPI_{GPI_iter}.csv')
            save_sa_counts_to_csv(sa_counts, sa_counts_filename)
            
            # Restart the counts and returns for the next iteration of policy evalutation
            sa_counts = defaultdict(lambda: 0)
            sa_returns = defaultdict(lambda: 0)
            
            # Update epsilon for the next iteration of policy evaluation
            epsilon = epsilon_decay('linear', GPI_iter, CONFIG["num_GPI_iter"], epsilon_start, CONFIG["epsilon_min"])
            market_params[3]['sellers'][1][2]['epsilon'] = epsilon
            print(f"New epsilon: {epsilon}")
            
            
            
        if episode % CONFIG["eps_per_evaluation"] == 0:
            GPI_iter += 1
            print(f"Starting GPI iteration {GPI_iter}")
            
    return  

   
# This takes a file name for a CSV file, containing the episode data, state,action,reward. It reads this file converts to lists
def load_episode_data(file: str) -> Tuple[List[Tuple], List[float], List[float]]:
    obs_list, action_list, reward_list = [], [], []

    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row_number, row in enumerate(reader, start=1):
            try:
                # Safely evaluate the Observation string to a tuple
                obs = ast.literal_eval(row[0])
                if not isinstance(obs, tuple):
                    raise ValueError(f"Observation at row {row_number} is not a tuple.")

                obs_list.append(obs)

                # Convert Action and Reward to float
                action_list.append(float(row[1]))
                reward_list.append(float(row[2]))
            except Exception as e:
                print(f"Error processing row {row_number}: {e}")
                continue  # Skip problematic rows

    return obs_list, action_list, reward_list


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

# takes in the counts and returns spits out the estimate for Q_new as a dictionary  
def average(sa_counts, sa_returns, save=False) -> Dict :
    # Ensure all keys in sa_returns are in sa_counts to avoid KeyError
    for key in sa_returns:
        if key not in sa_counts:
            raise KeyError(f"Key {key} found in sa_returns but not in sa_counts.")
        if sa_counts[key] == 0:
            raise ValueError(f"Count for key {key} is zero, cannot divide by zero.")

    # Create a new dictionary with the results of the division
    sa_average = {key: sa_returns[key] / sa_counts[key] for key in sa_returns}
    
    # count the percentage of returns that are zero, and then print that number
    # total_entries = len(sa_average)
    # zero_entries = sum(1 for value in sa_average.values() if value == 0)
    # zero_percentage = (zero_entries / total_entries) * 100
    # print(f"Percentage of zero values: {zero_percentage:.2f}%")

        
    return sa_average


def incremental_update(Q_new: Dict, Q_old: Dict, alpha) -> Dict:

    # Initialize the merged dictionary
    next_q_table = {}
    
    # Get all unique keys from both dictionaries
    all_keys = set(Q_new.keys()).union(set(Q_old.keys()))
    
    for key in all_keys:
        if key in Q_new and key in Q_old:
            # If the key exists in both, apply the update rule
            next_q_table[key] = Q_old[key] + alpha * (Q_new[key] - Q_old[key])
        elif key in Q_new:
            # If the key is only in Q_new, take its value from Q_new
            next_q_table[key] = Q_new[key]
        else:
            # If the key is only in Q_old, take its value from Q_old
            next_q_table[key] = Q_old[key]
    
    return next_q_table


def save_dict_to_csv(new_q_table : Dict) :
    
    sorted_new_q_table = sorted(new_q_table.items(), key=lambda x: x[0][0])
    
    with open('q_table_seller.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['State', 'Action', 'Q-Value'])
        # Write the data
        for (state, action), q_value in sorted_new_q_table:
            writer.writerow([state, action, q_value])
