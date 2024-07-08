import numpy as np 
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from q_table_data import load_q_table, dump_q_table


gamma = 1.0
alpha = 1e-4


def q_learn(
        obs_list: List[int], actions_list: List[int], rewards_list: List[float], n_obs_list: List[int]
    ) -> float:
        """Implements the Q-Learning method and updates 
        the Q-table based on agent experience.
        
        :param obs_list (List[int]): list of observations representing the environmental states
        :param actions_list (List[int]): list of indices of applied actions
        :param rewards_list (List[float]): list of received rewards
        :param n_obs_list (List[int]): list of observations representing the next environmental states
        :param q_table (DefaultDict): the Q-table to update
        :return (DefaultDict): updated Q-table
        """
        # Load the current q_table from the CSV file
        try:
            if type == 'Buyer':
                q_table = load_q_table('qlearning_q_table_buyer.csv')
            elif type == 'Seller':
                q_table = load_q_table('qlearning_q_table_seller.csv')
        except:
            q_table = defaultdict(lambda: 0)


        traj_length = len(rewards_list)
        num_actions = 500

        for t in range(traj_length):
            obs = obs_list[t]
            action = actions_list[t]
            reward = rewards_list[t]
            n_obs = n_obs_list[t]

            # Best action for the next state
            best_next_action = max(range(num_actions), key=lambda a: q_table[(n_obs, a)])

            # Update Q-value using Q-learning update rule
            q_table[(obs, action)] += alpha * (reward + gamma * q_table[(n_obs, best_next_action)] - q_table[(obs, action)])

        # Save the updated q_table back to the CSV file
        if type == 'Buyer':
            dump_q_table(q_table, 'qlearning_q_table_buyer.csv')
        elif type == 'Seller':
            dump_q_table(q_table, 'qlearning_q_table_seller.csv')

        return q_table
