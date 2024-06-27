import time
import matplotlib.pyplot as plt
import numpy as np


def evaluate(env, agent, max_steps, eval_episodes):
    """
    Evaluate configuration on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param agent (Agent): agent to act in environment
    :param max_steps (int): max number of steps per evaluation episode
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, int): mean of returns received over episodes and number of negative
        return evaluation, episodes
    """
    episodic_returns = []
    for eps_num in range(eval_episodes):
        obs, _ = env.reset()
        episodic_return = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            act = agent.act(obs)
            n_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated

            episodic_return += reward
            steps += 1

            obs = n_obs

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns)
    negative_returns = sum([ret < 0 for ret in episodic_returns])

    return mean_return, negative_returns
