import numpy as np
import random
import csv
import gymnasium as gym
from gymnasium import Space
from gymnasium import spaces
import torch
from torch import nn, Tensor
from torch.optim import Adam
from typing import Dict, Iterable, List, DefaultDict
from collections import defaultdict

from neural_network import Network
from BSE import RLAgent, Order

class Reinforce(RLAgent):
    def __init__(
            self,
            ttype, 
            tid, 
            balance, 
            params, 
            time, 
            action_space: spaces.Space, 
            obs_space: spaces.Space, 
            learning_rate,
            gamma=1.0, 
            epsilon=0.9,
    ):
        
        super().__init__(ttype, tid, balance, params, time, action_space, obs_space, gamma, epsilon)
        state_size = np.prod(obs_space.shape)
        action_size = action_space.n
        self.learning_rate = learning_rate
        
        # Create low and high bounds with padding for variable length lists
        low = np.array([[self.min_price, 1]] * self.max_length + [[0, 0]] * (self.max_length - 1))
        high = np.array([[self.max_price, self.num_traders]] * self.max_length + [[0, 0]] * (self.max_length - 1))

        self.observation_space = spaces.Dict(
            {'Bids': spaces.Box(
                    low=low,
                    high=high,
                    dtype=np.float32),
            'Asks': spaces.Box(
                    low=low,
                    high=high,
                    dtype=np.float32)
            })

        self.policy = Network(
            dims=(state_size, action_size), output_activation=torch.nn.modules.activation.Softmax
            )
        
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)


    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            order = None

        else:

            if self.type == 'Buyer':
                file = 'episode_buyer.csv'
            elif self.type == 'Seller':
                file = 'episode_seller.csv'
            
            # Write the current state (lob), action and reward
            reward = 0.0
            with open(file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([lob, 0.0, reward])
            order_type = self.orders[0].otype
            
            obs = self.current_obs
            state = torch.tensor(obs, dtype=torch.float32)
            action_prob = self.policy(state)

            # Explore - sample a random action
            if random.uniform(0, 1) < self.epsilon:
                if self.type == 'Buyer':
                    action = np.random.choice(self.action_space.n)
                    quote = self.orders[0].price * (1 - action)
                elif self.type == 'Seller':
                    action = np.random.choice(self.action_space.n)
                    quote = self.orders[0].price * (1 + action)

            # Exploit - choose the action with the highest probability
            else:
                if self.type == 'Buyer':
                    action = torch.argmax(action_prob).item()
                    quote = self.orders[0].price * (1 - action)
                elif self.type == 'Seller':
                    action = torch.argmax(action_prob).item()
                    quote = self.orders[0].price * (1 + action)

            # Check if it's a bad bid
            if self.type == 'Buyer' and quote > self.orders[0].price:
                quote = self.orders[0].price
            
            # Check if it's a bad ask
            elif self.type == 'Seller' and quote < self.orders[0].price:
                quote = self.orders[0].price

            order = Order(self.tid, order_type, (quote), self.orders[0].qty, time, lob['QID'])

            

        return order
    

    def update(
        self, observations: List[np.ndarray], actions: List[int], rewards: List[float],
        ) -> Dict[str, float]:
        # Initialise loss and returns
        p_loss = 0
        G = 0
        traj_length = len(observations)

        # Compute action probabilities using the current policy
        action_probabilities = self.policy(torch.tensor(observations, dtype=torch.float32))

        # Loop backwards in the episode
        for t in range(traj_length - 2, -1, -1):
            G = self.gamma * G + rewards[t+1]
            action_prob = action_probabilities[t, actions[t]]   # Probability of the action at time step t
            p_loss = p_loss - G * torch.log(action_prob)   # Minimise loss function

        p_loss = p_loss/traj_length   # Normalise policy loss

        # Backpropogate and perform optimisation step
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        return {"p_loss": float(p_loss)}