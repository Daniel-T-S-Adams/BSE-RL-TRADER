import random
import csv
import numpy as np 
import ast
from tqdm import tqdm
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from q_table_data import load_q_table, dump_q_table
from epsilon_scheduling import epsilon_decay

import torch
from torch import nn, Tensor
from torch.optim import Adam

from neural_network import Network

gamma = 1.0


def load_episode_data(file: str) -> Tuple[List[np.ndarray], List[float], List[float]]:
    obs_list, action_list, reward_list = [], [], []

    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            obs_str = row[0].strip()
            obs_str = obs_str.replace('"', '')  # Remove quotes
            obs_str = obs_str.replace('[', '').replace(']', '')  # Remove brackets

            # Convert the string to a numpy array and reshape
            obs_array = np.fromstring(obs_str, sep=' ').reshape((2, 10, 2))

            obs_list.append(obs_array)

            # Convert the Action and Reward to floats
            action_list.append(float(row[1]))
            reward_list.append(float(row[2]))

    return obs_list, action_list, reward_list


def update(
        observations: List[np.ndarray], actions: List[int], rewards: List[float],
        ) -> Dict[str, float]:
        # Initialise loss and returns
        p_loss = 0
        G = 0
        traj_length = len(observations)
        
        # Flatten each observation
        flattened_observations = [obs.flatten() for obs in observations]
        
        # Convert observations to a tensor
        obs_tensor = torch.tensor(flattened_observations, dtype=torch.float32)

        # Compute action probabilities using the current policy
        action_probabilities = policy(obs_tensor)
        
        # Loop backwards in the episode
        for t in range(traj_length - 2, -1, -1):
            G = gamma * G + rewards[t+1]
            action_prob = action_probabilities[t, int(actions[t])]   # Probability of the action at time step t
            p_loss = p_loss - G * torch.log(action_prob)   # Minimise loss function

        p_loss = p_loss/traj_length   # Normalise policy loss
        
        # Backpropogate and perform optimisation step
        policy_optim.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)       # Gradient clipping
        policy_optim.step()
        
        return {"p_loss": float(p_loss)}


def train(total_eps: int, market_params: tuple, eval_freq: int, epsilon) -> DefaultDict:
    # Dictionary to store training statistics
    stats = defaultdict(list)

    for episode in range(1, total_eps + 1):
        # Run a market session to generate the episode data
        market_session(*market_params)

        try:
            file = 'episode_seller.csv'
            obs_list, action_list, reward_list = load_episode_data(file)

            # Run the REINFORCE algorithm
            update_results = update(obs_list, action_list, reward_list)
            
            # Store the update results
            for key, value in update_results.items():
                stats[key].append(value)

        except Exception as e:
            print(f"Error processing seller episode {episode}: {e}")

        # Evaluate the policy at specified intervals
        if episode % eval_freq == 0:
            print(f"Episode {episode}: {update_results}")
            
            mean_return_seller, mean_return_list = evaluate(
            episodes=CONFIG['eval_episodes'], market_params=market_params, 
            policy=policy, file='episode_seller.csv')

            tqdm.write(f"EVALUATION: EP {episode} - MEAN RETURN SELLER {mean_return_seller}")

    return stats


def evaluate(episodes: int, market_params: tuple, policy, file) -> float:
    total_return = 0.0
    mean_return_list = []

    updated_market_params = list(market_params)    
    updated_market_params[3]['sellers'][4][2]['policy'] = policy
    updated_market_params[3]['sellers'][4][2]['epsilon'] = 0.0              # No exploring

    for _ in range(episodes):
        balance = 0.0
        market_session(*market_params)

        # Read the episode file
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                reward = float(row[2])
                balance += reward

        # Profit made by the RL agent at the end of the trading window
        total_return += balance
    
    mean_return = total_return / episodes
    mean_return_list.append(mean_return)

    return mean_return, mean_return_list
     


policy = Network(
     dims=(40, 32, 21), output_activation=nn.Softmax(dim=-1)
     )
        
policy_optim = Adam(policy.parameters(), lr=0.01, eps=1e-3)


CONFIG = {
    "total_eps": 1000,
    "eval_freq": 100,
    "eval_episodes": 100,
    "gamma": 1.0,
    "epsilon": 1.0,
}

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 60.0

sellers_spec = [('SHVR', 1), ('GVWY', 1), ('ZIC', 1), ('ZIP', 1), ('REINFORCE', 1, {'epsilon':1.0, 'policy': policy})]
buyers_spec = [('SHVR', 1), ('GVWY', 1), ('ZIC', 1), ('ZIP', 1)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

range1 = (50, 150)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (50, 150)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 15

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-poisson'}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': False, 'dump_tape': False, 'dump_blotters': False}
verbose = False

# Train the agent
training_stats = train(CONFIG['total_eps'],
                       market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
                       eval_freq=CONFIG['eval_freq'],
                       epsilon=CONFIG['epsilon']
                       )
