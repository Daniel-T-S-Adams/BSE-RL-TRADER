import random
import csv
import numpy as np 
from tqdm import tqdm
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from q_table_data import load_q_table, dump_q_table, update_q_table
from epsilon_scheduling import epsilon_decay
#file handling
from typing import List, Tuple
import shutil
import csv



gamma = 0.3 # discounting factor

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


def learn(obs: List[int], actions: List[int], rewards: List[float], type, sa_counts, sa_rewards) -> Tuple[DefaultDict, DefaultDict]:

    if len(rewards) == 0:
        return sa_counts, sa_rewards

    # Precompute returns G for every timestep
    traj_length = len(rewards)
    if traj_length == 0:
        raise ValueError("The rewards list is empty")
    G = [ 0 for n in range(traj_length) ]
    G[-1] = rewards[-1]
    for t in range(traj_length - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1] 
    
    # Update rewards and counts 
    for t in range(traj_length):
        state_action_pair = (tuple(obs[t]), actions[t])
        sa_counts[state_action_pair] += 1
        sa_rewards[state_action_pair] += G[t]
        
    
    return sa_counts, sa_rewards


def average(sa_counts, sa_rewards, save=False):
    # Ensure all keys in sa_rewards are in sa_counts to avoid KeyError
    for key in sa_rewards:
        if key not in sa_counts:
            raise KeyError(f"Key {key} found in sa_rewards but not in sa_counts.")
        if sa_counts[key] == 0:
            raise ValueError(f"Count for key {key} is zero, cannot divide by zero.")

    # Create a new dictionary with the results of the division
    sa_average = {key: sa_rewards[key] / sa_counts[key] for key in sa_rewards}
    
    # Sort the dictionary by state
    sorted_sa_average = sorted(sa_average.items(), key=lambda x: x[0][0])
    
    # count the percentage of rewards that are zero, and then print that number
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

def evaluate(episodes: int, market_params: tuple, q_table: DefaultDict, file, new_epsilon) -> float:

    updated_market_params = list(market_params)    
    # if file == 'q_table_buyer.csv':
    #     updated_market_params[3]['buyers'][0][2]['q_table_buyer'] = 'q_table_buyer.csv'
    #     updated_market_params[3]['buyers'][0][2]['epsilon'] = 0.0                           # No exploring
    # elif file == 'q_table_seller.csv':
    updated_market_params[3]['sellers'][1][2]['q_table_seller'] = 'q_table_seller.csv'
    updated_market_params[3]['sellers'][1][2]['epsilon'] = new_epsilon                          # No exploring

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


def train(total_eps: int, market_params: tuple, eval_freq: int, epsilon_start: float) -> DefaultDict:
    GPI_iter = 0
    new_epsilon = epsilon_start
    print(f"Starting GPI iteration {GPI_iter}")
    sa_counts = defaultdict(lambda: 0)
    sa_rewards = defaultdict(lambda: 0)
    for episode in range(1, total_eps + 1):
        new_MC_iteration = False
        market_session(*market_params)
        
        # Check if there's a sell trader
        try:
            file = 'episode_seller.csv'
            obs_list, action_list, reward_list = load_episode_data(file)
        except Exception as e:
            print(f"Error loading seller episode {episode}: {e}")
            pass
        
            # update the count and rewards
        try:
            sa_counts, sa_rewards = learn(obs_list, action_list, reward_list, 'Seller', sa_counts, sa_rewards)
        except Exception as e:
            print(f"Error computing new count and rewards for seller episode {episode}: {e}")
            pass

        if episode % M == 0: 
            # Save sa_counts to a CSV file with the episode number in the filename
            sa_counts_filename = f'sa_counts_episode_{episode}.csv'
            save_sa_counts_to_csv(sa_counts, sa_counts_filename)
            # divide the q_table by number of times each state visited and save it to a csv file name q_table_seller.csv
            save = True
            average(sa_counts, sa_rewards, save)
            market_params[3]['sellers'][1][2]['q_table_seller'] = 'q_table_seller.csv'
            # save the sellers q_table file for the past Monte Carlo iterations
            new_file_name = f'q_table_seller_after_episode_{episode}.csv'
            shutil.copy('q_table_seller.csv', new_file_name)  
            # restart the counts and rewards for the next Monte Carlo iterations
            sa_counts = defaultdict(lambda: 0)
            sa_rewards = defaultdict(lambda: 0)
            GPI_iter += 1
            print(f"Starting GPI iteration {GPI_iter}")
            new_MC_iteration = True
        
        
        # Get some file logs every now and again
        if episode % 2000 == 0:
            # save the sellers dictionary
            # new_file_name = f'q_sa_count_episode_num_{episode}.csv'
            # save_dict_to_csv(new_file_name, sa_counts)
            # new_file_name = f'q_sa_reward_episode_num_{episode}.csv'
            # save_dict_to_csv(new_file_name, sa_rewards)
            # Save the sellers q_table file
            # new_file_name = f'q_table_seller_episode_num_{episode}.csv'
            # shutil.copy('q_table_seller.csv', new_file_name)
            # # save the sellers episode file
            # new_file_name = f'episode_seller_episode_num_{episode}.csv'
            # shutil.copy('episode_seller.csv', new_file_name) 
            # see if q_table is converging
            if not new_MC_iteration:
                save = False
                current_qtable = average(sa_counts, sa_rewards, save)
                if 'previous_qtable' in locals():
                    total_difference = 0
                    
                    for key in current_qtable:
                        
                        previous_value = previous_qtable.get(key, 0)
                        total_difference += abs(current_qtable[key] - previous_value)
                    
                    # Print or log the total difference
                    print(f"Total difference after episode {episode}: {total_difference}")
                    
                # save q_table
                previous_qtable = current_qtable
            
        
        # Perform evaluation every `eval_freq` episodes
        if episode % eval_freq == 0:
            print(f"Evaluating after Training Episode {episode}/{total_eps}")
            
            
            cumulative_stats = evaluate(
            episodes=CONFIG['eval_episodes'], market_params=market_params, 
            q_table='q_table_seller.csv', file='episode_seller.csv', new_epsilon = new_epsilon)
            
            
            for ttype in cumulative_stats:
                print(f"EVALUATION: EP {episode}, {ttype} average profit: {cumulative_stats[ttype]['avg_profit']}")
                
                
            # update epsilon 
            new_epsilon = epsilon_decay('linear', GPI_iter, number_of_policy_improvements, epsilon_start, 0.05)
            market_params[3]['sellers'][1][2]['epsilon'] = new_epsilon
            print(f"New epsilon: {new_epsilon}")
            
    return 5

M = 300 # number of episodes for monte carlo
number_of_policy_improvements = 10
CONFIG = {
    "total_eps": number_of_policy_improvements*M,
    "eval_freq": M,
    "eval_episodes": 2000,
    "gamma": 0.0,
    "epsilon": 0.5,
}

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 30.0

sellers_spec = [('GVWY',19), ('RL', 1, {'epsilon': CONFIG['epsilon']})]
buyers_spec = [('GVWY',20)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

range1 = (1, 3)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (1, 3)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 30

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-fixed'}

dump_flags = {'dump_strats': True, 'dump_lobs': True, 'dump_avgbals': True, 'dump_tape': True, 'dump_blotters': True}
verbose = False


# Training the RL agent with evaluation
q_table = train(total_eps=CONFIG['total_eps'], 
                market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
                eval_freq=CONFIG['eval_freq'],
                epsilon_start=CONFIG['epsilon'])


print(f"Finished with gamma equal to {gamma}")
