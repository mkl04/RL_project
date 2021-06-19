import os, random
import numpy as np
import torch
from torch import nn
import itertools
from baselines_wrappers import DummyVecEnv
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
import time
import argparse

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

from train import nature_cnn, Network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('-game', help='options: tennis, space', default="space")
    parser.add_argument('-agent', help='options: dqn, double', default="dqn")
    parser.add_argument('-num_test', help='number of tests', default=3, type=int)
    args = parser.parse_args()

    if args.game == "tennis":
        atari_name = "Tennis"
        NUM_LIVES = 1
    if args.game == "space":
        atari_name = "SpaceInvaders"
        NUM_LIVES = 3

    make_env = lambda: make_atari_deepmind(atari_name+'NoFrameskip-v4', scale_values=True, clip_rewards=False,
                                            max_episode_steps=60000) # 5s <> 1000 steps -> 5min <> 60000 steps

    vec_env = DummyVecEnv([make_env for _ in range(1)])
    env = BatchedPytorchFrameStack(vec_env, k=4)

    net = Network(env, device)
    net = net.to(device)
    net.load('./models/{}_{}.pack'.format(args.game, args.agent))

    NUM_TEST = args.num_test

    sum_rewards = 0
    lives=NUM_LIVES
    list_rewards_games = []
    for e in range(NUM_TEST):

        obs = env.reset()
        beginning_episode = True

        start_time = time.time()

        print("================================================")
        print("GAME ", e+1)
        
        for t in range(1000000000000):

            if isinstance(obs[0], PytorchLazyFrames):
                act_obs = np.stack([o.get_frames() for o in obs])
                action = net.act(act_obs, 0.05)
            else:
                action = net.act(obs, 0.05)

            if beginning_episode:
                action = [1]
                beginning_episode = False

            obs, rew, done, _ = env.step(action)

            sum_rewards+=rew[0]
            tmp = time.time() - start_time

            if done[0]:
                lives-=1
                print("Game {} -- remaining lives: {}".format(e+1, lives) )
                
                if lives==0:
                    print("Game {} ends  -- Total Reward {} -- time: {}  -- t {}".format(e+1, sum_rewards, tmp, t+1))
                    list_rewards_games.append(sum_rewards)
                    sum_rewards = 0
                    lives = NUM_LIVES
                    break

    print(list_rewards_games)
    print( "MEAN REWARDS: {} +- {}".format(  np.mean(list_rewards_games),  np.std(list_rewards_games)  ) )