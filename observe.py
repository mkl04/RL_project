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
    args = parser.parse_args()

    if args.game == "tennis":
        atari_name = "Tennis"
    if args.game == "space":
        atari_name = "SpaceInvaders"

    make_env = lambda: make_atari_deepmind(atari_name+'NoFrameskip-v4', scale_values=True)

    vec_env = DummyVecEnv([make_env for _ in range(1)])

    env = BatchedPytorchFrameStack(vec_env, k=4)

    net = Network(env, device)
    net = net.to(device)
    net.load('./models/{}_{}.pack'.format(args.game, args.agent))

    obs = env.reset()
    beginning_episode = True
    for t in itertools.count():
        if isinstance(obs[0], PytorchLazyFrames):
            act_obs = np.stack([o.get_frames() for o in obs])
            action = net.act(act_obs, 0.0)
        else:
            action = net.act(obs, 0.0)

        if beginning_episode:
            action = [1]
            beginning_episode = False

        obs, rew, done, _ = env.step(action)
        env.render()
        time.sleep(0.02)

        if done[0]:
            obs = env.reset()
            beginning_episode = True