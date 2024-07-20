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
import torch.nn.functional as F

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
        v_loss = 0
        G = 0
        traj_length = len(observations)

        # Normalize rewards
        rewards = np.array(rewards)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards) + 1e-10  # Add a small value to avoid division by zero
        normalised_rewards = (rewards - reward_mean) / reward_std
    
        # Flatten each observation
        flattened_observations = [obs.flatten() for obs in observations]
        
        # Convert observations to a tensor
        obs_tensor = torch.tensor(flattened_observations, dtype=torch.float32)

        # Compute action probabilities using the current policy
        eps = 1e-10
        action_probabilities = policy_net(obs_tensor) + eps
        baseline_values = value_net(obs_tensor).squeeze()
        
        # Precompute returns G for every timestep
        G = [ 0 for n in range(traj_length) ]
        G[-1] = normalised_rewards[-1]
        for t in range(traj_length - 2, -1, -1):
            G[t] = normalised_rewards[t] + gamma * G[t + 1]

        G = torch.tensor(G, dtype=torch.float32)
        advantage = G - baseline_values
        p_loss = torch.mean(-advantage * torch.log(action_probabilities[torch.arange(traj_length), actions]))
        v_loss = F.mse_loss(baseline_values, G)

        # Backpropogate and perform optimisation step for the policy
        policy_optim.zero_grad()
        p_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)       # Gradient clipping
        policy_optim.step()

        # Backpropogate and perform optimisation step for the value function
        value_optim.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)       # Gradient clipping
        value_optim.step()
        
        return {"p_loss": float(p_loss), "v_loss": float(v_loss)}


def train(total_eps: int, market_params: tuple, eval_freq: int, epsilon) -> DefaultDict:
    # Dictionary to store training statistics
    stats = defaultdict(list)
    mean_return_list = []

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
            
            mean_return_seller = evaluate(
                episodes=CONFIG['eval_episodes'], market_params=market_params, 
                policy=policy_net, file='episode_seller.csv')
            tqdm.write(f"EVALUATION: EP {episode} - MEAN RETURN SELLER {mean_return_seller}")
            mean_return_list.append(mean_return_seller)

    return stats, mean_return_list


def evaluate(episodes: int, market_params: tuple, policy, file) -> float:
    total_return = 0.0

    updated_market_params = list(market_params)    
    updated_market_params[3]['sellers'][1][2]['policy'] = policy
    updated_market_params[3]['sellers'][1][2]['epsilon'] = 0.0         # No exploring

    for _ in range(episodes):
        balance = 0.0
        market_session(*updated_market_params)

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

    return mean_return
     

policy_net = Network(
    dims=(40, 32, 21), output_activation=nn.Softmax(dim=-1)
    )

value_net = Network(dims=(40, 32, 32, 1), output_activation=None)
        
policy_optim = Adam(policy_net.parameters(), lr=1e-4, eps=1e-3)
value_optim = Adam(value_net.parameters(), lr=1e-4, eps=1e-3)


CONFIG = {
    "total_eps": 50000,
    "eval_freq": 2500,
    "eval_episodes": 1000,
    "gamma": 1.0,
    "epsilon": 1.0,
}

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 60.0

sellers_spec = [('GVWY', 9), ('REINFORCE', 1, {'epsilon': 1.0, 'policy': policy_net})]
buyers_spec = [('GVWY', 10)]

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
training_stats, eval_returns_list = train(CONFIG['total_eps'],
                                    market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
                                    eval_freq=CONFIG['eval_freq'],
                                    epsilon=CONFIG['epsilon']
                                    )


policy_loss = training_stats['p_loss']
plt.plot(policy_loss, linewidth=1.0)
plt.title("Policy Loss vs Episode")
plt.xlabel("Episode number")
# plt.savefig("policy_loss.png")
# plt.close()
plt.show()

value_loss = training_stats['v_loss']
plt.plot(value_loss, linewidth=1.0)
plt.title("Value Loss vs Episode")
plt.xlabel("Episode number")
# plt.savefig("value_loss.png")
# plt.close()
plt.show()

x_ticks = np.arange(CONFIG['eval_freq'], CONFIG['total_eps']+1, CONFIG['eval_freq'])
plt.plot(x_ticks, eval_returns_list, linewidth=1.0)
plt.title("Mean returns - REINFORCE")
plt.xlabel("Episode number")
# plt.savefig("mean_returns.png")
plt.show()