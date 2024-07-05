import numpy as np 
import random
from typing import List, Dict, DefaultDict
from collections import defaultdict
from BSE import market_session
from tqdm import tqdm
import csv

gamma = 1.0

def load_q_table(file_path: str) -> DefaultDict:
    q_table = defaultdict(lambda: 0)
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                state, action, q_value = row
                q_table[(state, int(float(action)))] = float(q_value)
    except FileNotFoundError:
        pass  # If the file does not exist, return an empty q_table
    return q_table


def dump_action_values(q_table: DefaultDict, file_path: str):
    """
    Save the Q-table to a CSV file, updating existing entries or adding new ones.

    :param q_table (DefaultDict): The Q-table to save.
    :param file_path (str): The path to the file where the Q-table will be saved.
    """
    # Load the existing Q-table from the file
    existing_q_table = load_q_table(file_path)

    # Update existing Q-table with new entries
    for (state, action), q_value in q_table.items():
        existing_q_table[(state, action)] = q_value

    # Write the updated Q-table back to the file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['State', 'Action', 'Q-Value'])  # Write the header
        for (state, action), q_value in existing_q_table.items():
            writer.writerow([state, action, q_value])
            # state_str = ','.join(map(str, state))  # Convert state tuple to string
            # writer.writerow([f'({state_str})', action, q_value])


def learn(obs: List[int], actions: List[int], rewards: List[float]) -> Dict:
        # Load the current q_table from the CSV file
        q_table = load_q_table('q_table.csv')

        sa_counts = {}
        q_table = defaultdict(lambda: 0)
        traj_length = len(rewards)
        G = 0
        state_action_list = list(zip([tuple(o) for o in obs], actions))
        
        # Iterate over the trajectory backwards
        for t in range(traj_length - 1, -1, -1):
            state_action_pair = (tuple(obs[t]), actions[t])

            # Check if this is the first visit to the state-action pair
            if state_action_pair not in state_action_list[:t]:
                G = gamma*G + rewards[t]

                # Monte-Carlo update rule
                sa_counts[state_action_pair] = sa_counts.get(state_action_pair, 0) + 1
                q_table[state_action_pair] += (
                    G - q_table[state_action_pair]
                    ) / sa_counts.get(state_action_pair, 0)
                
                # updated_values[state_action_pair] = q_table[state_action_pair]

         # Save the updated q_table back to the CSV file
        dump_action_values(q_table, 'q_table.csv')
      
        return q_table


def evaluate(episodes: int, market_params: tuple, q_table: DefaultDict) -> float:
    total_return = 0.0
    
    for _ in range(episodes):
        obs_list, action_list, reward_list = market_session(*market_params)
        # Compute total return for the current evaluation episode
        total_return += sum(reward_list)
    
    mean_return = total_return / episodes
    return mean_return

# Should change epsilon to 0.0 here so the agent doesn't explore
# but rather just tries to maximise rewards
def train(episodes: int, market_params: tuple, eval_freq: int) -> DefaultDict:
    for episode in range(1, episodes + 1):
        # Run one market session to get observations, actions, and rewards
        obs_list, action_list, reward_list = market_session(*market_params)

        # Learn from the experience
        q_table = learn(obs_list, action_list, reward_list)
        
        # Perform evaluation every `eval_freq` episodes
        if episode % eval_freq == 0:
            print(f"Training Episode {episode}/{episodes}")
            mean_return = evaluate(episodes=100, market_params=market_params, q_table=q_table)
            tqdm.write(f"EVALUATION: EP {episode} - MEAN RETURN {mean_return}")

    return q_table


# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 100.0

buyers_spec = [('SHVR', 5), ('GVWY', 5), ('ZIC', 5), ('ZIP', 5), ('RL', 1, {})]
sellers_spec = [('SHVR', 5), ('GVWY', 5), ('ZIC', 5), ('ZIP', 5)]

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


# Training the RL agent with evaluation every 10 episodes
q_table = train(episodes=100, market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), eval_freq=10)