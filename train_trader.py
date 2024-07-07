import numpy as np 
import random
from typing import List, Dict, DefaultDict, Tuple
from collections import defaultdict
from BSE import market_session
from tqdm import tqdm
import csv
from matplotlib import pyplot as plt


gamma = 1.0
alpha = 1e-4


def load_episode_data(file: str) -> Tuple[List, List, List]:
    obs_list, action_list, reward_list = [], [], []

    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            obs_list.append(np.fromstring(row[0].strip('[]'), sep=' '))
            action_list.append(int(float(row[1])))
            reward_list.append(float(row[2]))

    return obs_list, action_list, reward_list


def epsilon_decay(strat, epsilon, timestep, max_timestep, eps_start=1.0, eps_min=0.05, eps_decay=0.01):

    if strat == 'constant':
        epsilon = 0.9

    if strat == 'linear':
        decay_steps = eps_decay * max_timestep
        epsilon = max(eps_min, eps_start - (eps_start - eps_min) * min(1.0, timestep / decay_steps))

    if strat == 'exponential':
        epsilon = max(eps_min, eps_decay*epsilon)

    return epsilon


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
    dump_q_table(q_table, 'q_table.csv')
    
    return q_table


def q_learn(
        obs: int, action: int, reward: float, n_obs: int, done: bool
    ) -> float:
        """Implements the Q-Learning method and updates 
        the Q-table based on agent experience.

        :param obs (int): received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (int): received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
        # Load the current q_table from the CSV file
        q_table = load_q_table('q_table.csv')

        num_actions = 500

        # Best action for the next state
        a_ = np.argmax([q_table[(n_obs, a)] for a in range(num_actions)])

        # Update Q-value using Q-learning update rule
        q_table[(obs, action)] += (
            alpha * (reward + gamma * q_table[(n_obs, a_)] * (1 - done) - q_table[(obs, action)])
            )

        obs = n_obs



def evaluate(episodes: int, market_params: tuple, q_table: DefaultDict) -> float:
    total_return = 0.0
    mean_return_list = []
    
    for _ in range(episodes):
        balance = 0.0
        market_session(*market_params)

        # Read the episode.csv file
        with open('episode.csv', 'r') as f:
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

# Should change epsilon to 0.0 here so the agent doesn't explore
# but rather just tries to maximise rewards
def train(episodes: int, market_params: tuple, eval_freq: int, epsilon) -> DefaultDict:
    for episode in range(1, episodes + 1):

        market_session(*market_params)
        
        # # Update market_params to include the current epsilon
        # updated_market_params = list(market_params)
        # if updated_market_params[3]['buyers'][4][0] == 'RL':
        #     updated_market_params[3]['buyers'][4][2]['epsilon'] = epsilon
        
        # epsilon = epsilon_decay('linear', epsilon, episode, CONFIG['total_eps'])

        # # Run one market session to get observations, actions, and rewards
        # market_session(*updated_market_params)

        file = 'episode.csv'
        obs_list, action_list, reward_list = load_episode_data(file)

        # Learn from the experience
        q_table = learn(obs_list, action_list, reward_list)
            
        # Perform evaluation every `eval_freq` episodes
        if episode % eval_freq == 0:
            print(f"Training Episode {episode}/{episodes}")
            mean_return, mean_return_list = evaluate(episodes=CONFIG['eval_episodes'], market_params=market_params, q_table=q_table)
            tqdm.write(f"EVALUATION: EP {episode} - MEAN RETURN {mean_return}")

    return q_table


CONFIG = {
    "total_eps": 10,
    "eval_freq": 100,
    "eval_episodes": 100,
    "gamma": 1.0,
    "epsilon": 1.0,
}

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 100.0

buyers_spec = [('SHVR', 5), ('GVWY', 5), ('ZIC', 5), ('ZIP', 5), ('RL', 1, {'q_table': 'q_table.csv', 'epsilon': CONFIG['epsilon']})]
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


# Training the RL agent with evaluation
q_table = train(episodes=CONFIG['total_eps'], 
                market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
                eval_freq=CONFIG['eval_freq'],
                epsilon=CONFIG['epsilon'])




