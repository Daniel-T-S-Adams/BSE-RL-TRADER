import random
import csv
import numpy as np 


from tqdm import tqdm
from BSE import market_session
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from epsilon_scheduling import linear_epsilon_decay
from converting_csv_and_dictionary import save_q_table_dict_to_csv, save_sa_counts_to_csv


import ast

# File handling
import shutil
import csv
import os

# Importing Global Parameters
from GlobalParameters import CONFIG

# Import the logging module and create a module-level logger
import logging
logger = logging.getLogger(__name__)

###### The functions

def train(total_eps: int, market_params: tuple, epsilon_start: float) -> DefaultDict:
    """
    Train the RL agent over a specified number of episodes.

    Parameters:
        total_eps (int): Total number of episodes to train.
        market_params (tuple): Parameters for the market session.
        epsilon_start (float): Starting value of epsilon for exploration.

    Returns:
        DefaultDict: A default dictionary containing training results.
    """
    ## Initialize everything: ##
    
    # Start the GPI iterations at 1 
    GPI_iter = 1
    logger.info(f"Starting GPI iteration {GPI_iter}")
    
    epsilon = epsilon_start
    
    # Initialize the counts and returns dictionaries
    sa_counts = defaultdict(lambda: 0)
    sa_returns = defaultdict(lambda: 0)
    
    # Empty dictionary 
    Q_old = defaultdict(lambda: 0.0)
    
    for episode in range(1, total_eps + 1):
        # Run the market session once, it returns a list of observations, actions and rewards for the RL trader 
        obs_list, action_list, reward_list = market_session(*market_params)
        
        # Update the count and returns
        try:
            sa_counts, sa_returns = learn(obs_list, action_list, reward_list, sa_counts, sa_returns)
        except Exception as e:
            logger.error(f"Error computing new count and returns for seller episode {episode}: {e}")
            pass

        # If we have done the designated number of episodes for this policy evaluation, compute the q_values i.e. q_table
        if episode % CONFIG["eps_per_evaluation"] == 0: 
            # Compute the Q_mc by averaging 
            Q_mc = average(sa_counts, sa_returns)
            # Compute the next q_table by an incremental update
            next_q_table = incremental_update(Q_mc, Q_old, CONFIG["alpha"])    
            # Save this Q_table for the next increment step
            Q_old = next_q_table

            # update the market parameter q_table dictionary with the new q_table
            market_params[3]['sellers'][1][2]['q_table_seller'] = next_q_table

            if episode % CONFIG["GPI_CSV_save_freq"] == 0:
                logger.info(f"Saving CSV files for GPI iter {GPI_iter}")
                #### Save CSV files (every so often?) (for later inspection)  ####
                # Save the new q_table dictionary to a CSV file named q_table_seller.csv
                q_table_file_name = os.path.join(CONFIG["q_tables"], f'q_table_seller_after_GPI_{GPI_iter}.csv')
                save_q_table_dict_to_csv(next_q_table, q_table_file_name) 
                
                # Save sa_counts to a CSV file for this GPI_iter (to keep track)
                sa_counts_filename = os.path.join(CONFIG["counts"], f'sa_counts_after_GPI_{GPI_iter}.csv')
                save_sa_counts_to_csv(sa_counts, sa_counts_filename)
                #### End of saving CSV files ####
            
            # Update epsilon for the next iteration of policy evaluation
            epsilon = linear_epsilon_decay(
                GPI_iter, 
                CONFIG["num_GPI_iter"], 
                epsilon_start, 
                CONFIG["epsilon_min"], 
                CONFIG["epsilon_decay"])
            market_params[3]['sellers'][1][2]['epsilon'] = epsilon
            logger.info(f"New epsilon: {epsilon}")
            

            # Restart the counts and returns for the next iteration of policy evaluation
            sa_counts = defaultdict(lambda: 0)
            sa_returns = defaultdict(lambda: 0)

            GPI_iter += 1
            logger.info(f"Starting GPI iteration {GPI_iter}")
    
    return  


def learn(
    obs: List[int], 
    actions: List[int], 
    rewards: List[float], 
    sa_counts, 
    sa_returns
) -> Tuple[DefaultDict, DefaultDict]:
    """
    Update the counts and returns for each state-action pair based on observed trajectories.

    Parameters:
        obs (List[int]): List of observations (states).
        actions (List[int]): List of actions taken.
        rewards (List[float]): List of rewards received.
        sa_counts (DefaultDict): State-action counts.
        sa_returns (DefaultDict): State-action returns.

    Returns:
        Tuple[DefaultDict, DefaultDict]: Updated state-action counts and returns.
    """
    traj_length = len(rewards)   
    
    if traj_length == 0:
        return sa_counts, sa_returns

    # Precompute returns G for every timestep
    G = [0 for _ in range(traj_length)]
    G[-1] = rewards[-1]
    for t in range(traj_length - 2, -1, -1):
        G[t] = rewards[t] + CONFIG["gamma"] * G[t + 1] 
    
    # Update returns and counts 
    for t in range(traj_length):
        state_action_pair = (tuple(obs[t]), actions[t])
        sa_counts[state_action_pair] += 1
        sa_returns[state_action_pair] += G[t]
        
    return sa_counts, sa_returns

# Takes in the counts and returns, spits out the estimate for Q_mc as a dictionary  
def average(sa_counts, sa_returns, save=False) -> Dict:
    """
    Calculate the average return for each state-action pair.

    Parameters:
        sa_counts (dict): State-action counts.
        sa_returns (dict): State-action returns.
        save (bool): Whether to save the averages to a file (unused).

    Returns:
        Dict: Average returns for each state-action pair.
    """
    # Ensure all keys in sa_returns are in sa_counts to avoid KeyError
    for key in sa_returns:
        if key not in sa_counts:
            raise KeyError(f"Key {key} found in sa_returns but not in sa_counts.")
        if sa_counts[key] == 0:
            raise ValueError(f"Count for key {key} is zero, cannot divide by zero.")

    
    # Create a new dictionary with the results of the division
    sa_average = {key: sa_returns[key] / sa_counts[key] for key in sa_returns}
    
    # Uncomment the following lines if you want to log the percentage of zero values
    # total_entries = len(sa_average)
    # zero_entries = sum(1 for value in sa_average.values() if value == 0)
    # zero_percentage = (zero_entries / total_entries) * 100
    # logger.info(f"Percentage of zero values: {zero_percentage:.2f}%")
        
    return sa_average

def incremental_update(Q_mc: Dict, Q_old: Dict, alpha) -> Dict:
    """
    Perform an incremental update of the Q-values.

    Parameters:
        Q_mc (Dict): The new Q-values.
        Q_old (Dict): The previous Q-values.
        alpha (float): Learning rate.

    Returns:
        Dict: Updated Q-values.
    """
    # Initialize the merged dictionary
    next_q_table = defaultdict(lambda: 0.0)
    
    # Get all unique keys from both dictionaries
    all_keys = set(Q_mc.keys()).union(set(Q_old.keys()))
    
    for key in all_keys:
        if key in Q_mc and key in Q_old:
            # If the key exists in both, apply the update rule
            next_q_table[key] = Q_old[key] + alpha * (Q_mc[key] - Q_old[key])
        elif key in Q_mc:
            # If the key is only in Q_mc, take its value from Q_mc
            next_q_table[key] = Q_mc[key]
        else:
            # If the key is only in Q_old, take its value from Q_old
            next_q_table[key] = Q_old[key]
    
    return next_q_table

