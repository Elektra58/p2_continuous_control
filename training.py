import json

import torch
from ddpg_agent import Agent
from collections import deque

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import Config, MyEncoder


def ddpg(agent, config):
    scores_window = deque(maxlen=config.window)
    scores = []
    ma_scores = []
    std_scores = []

    for i_episode in range(1, config.n_episodes + 1):
        env_info = config.env.reset(train_mode=True)[config.brain_name]
        states = env_info.vector_observations

        agent.reset()

        score = np.zeros(config.n_agents)

        for t in range(config.max_t):
            actions = agent.act(states)

            env_info = config.env.step(actions)[config.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states

            score += rewards

            if np.any(dones):
                break

        avg_score = np.mean(score)
        scores_window.append(avg_score)
        scores.append(avg_score)

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tScore: {avg_score:.3f}', end='')

        ma_score = np.mean(scores_window)
        ma_scores.append(ma_score)

        std_score = np.std(scores_window)
        std_scores.append(std_score)

        str_ = f'\rEpisode {i_episode:>4}\tCurrent Score {avg_score:5.2f}' + \
               f'\tAvg Score {ma_score:5.2f}\tStd Score {std_score:5.2f}'

        if i_episode % config.n_print == 0:
            print(str_)
        else:
            print(str_, end='')

        if ma_score >= config.target:
            str_ = f'\nEnvironment solved in {i_episode - config.window:d} episodes' + \
                   f'\tAverage Score: {ma_score:.2f}'
            print(str_)

            torch.save(agent.actor_local.state_dict(), f'results/model {config.model_id}/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), f'results/model {config.model_id}/checkpoint_critic.pth')

            config.target_episode = i_episode - config.window
            config.target_score = ma_score

            break

    return scores, ma_scores, std_scores