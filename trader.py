import numpy as np 
import random
import environment
from environment import AuctionEnv, lob
from evaluate import evaluate
from typing import List, Dict, DefaultDict
import gymnasium as gym
from gymnasium import Space
from gymnasium import spaces
from collections import defaultdict
from tqdm import tqdm

class Trader():
    def __init__(self, action_space: Space, obs_space: Space,
                 gamma: 0.0, epsilon: 0.9):
        self.action_space = action_space
        self.obs_space = obs_space
        self.num_actions = spaces.flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)
        self.sa_counts = {}

    # implement epsilon-greedy action selection
    def act(self, obs: int) -> int:
        obs = tuple(obs)
        if random.uniform(0, 1) < self.epsilon:
            # Exploit - choose the action with the highest probability
            return self.action_space.sample()
        else:
            # Explore - sample a random action
            return max(list(range(self.action_space.n)), key = lambda x: self.q_table[(obs, x)])


    def learn(self, obs: List[int], actions: List[int], rewards: List[float]) -> Dict:

        traj_length = len(rewards)
        G = 0
        state_action_list = list(zip(obs, actions))
        updated_values = {}
        
        # Iterate over the trajectory backwards
        for t in range(traj_length - 1, -1, -1):
            state_action_pair = (tuple(obs[t]), actions[t])

            # Check if this is the first visit to the state-action pair
            if state_action_pair not in state_action_list[:t]:
                G = self.gamma*G + rewards[t]

                # Monte-Carlo update rule
                self.sa_counts[state_action_pair] = self.sa_counts.get(state_action_pair, 0) + 1
                self.q_table[state_action_pair] += (
                    G - self.q_table[state_action_pair]
                    ) / self.sa_counts.get(state_action_pair, 0)
                
                updated_values[state_action_pair] = self.q_table[state_action_pair]
      
        return updated_values        


def monte_carlo_eval(
        env,
        config,
        q_table,
        render=False):
    """
    Evaluate configuration of MC on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = Trader(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=CONFIG["gamma"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    if render:
        eval_env = gym.make(CONFIG["env"], render_mode="human")
    else:
        eval_env = env
    return evaluate(eval_env, eval_agent, config["eval_eps_max_steps"], config["eval_episodes"])



def train(env, config):
    """
    Train and evaluate MC on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        returns over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table, final state-action counts
    """
    agent = Trader(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in tqdm(range(1, config["total_eps"] + 1)):
        obs, _ = env.reset()

        t = 0
        episodic_return = 0

        obs_list, act_list, rew_list = [], [], []
        while t < config["eps_max_steps"]:
            act = agent.act(obs)

            n_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated

            obs_list.append(obs)
            rew_list.append(reward)
            act_list.append(act)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        agent.learn(obs_list, act_list, rew_list)
        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = monte_carlo_eval(env, config, agent.q_table)
            tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table



CONFIG = {
    "env": 'Auction-v0',
    "eps_max_steps": 200,
    "eval_episodes": 500,
    "eval_eps_max_steps": 200,
    "eval_freq": 10000,
    "total_eps": 100000,
    "gamma": 0.0,
    "epsilon": 0.9,
}


# Register the gymnasium environment
gym.register(
    id='Auction-v0',
    entry_point='__main__:AuctionEnv',
)

if __name__ == "__main__":
    env = gym.make(CONFIG["env"], lob=lob)
    total_reward, _, _, q_table = train(env, CONFIG)