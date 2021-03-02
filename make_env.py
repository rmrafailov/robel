#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:21:07 2021

@author: rafael
"""

import tools
import wrappers
import gym
import robel
import d4rl
import pathlib
from stable_baselines3.common.monitor import Monitor
import numpy as np
import copy
import datetime
import io
import pathlib
import pickle
import re
import uuid

import gym
import h5py
import numpy as np

class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


config = AttrDict()
#config.task = 'd4rl_DClawTurnRandomDynamics-v0'
#config.task = 'd4rl_DClawScrewVelp4-v0'
#config.task = 'd4rl_DClawScrewVelp4-v1'
config.task = 'd4rl_DClawScrewVelp3-v0'
config.task = 'd4rl_DClawScrewVelp3-v1'
#config.task = 'd4rl_DClawScrewVel-v1'
config.dir = '/DClaw_task'

config.action_repeat = 2
config.time_limit = 200
config.im_size = 128
config.precision = 32
config.use_transform = False
config.pad = 6

def save_episodes(directory, episodes):
  directory = pathlib.Path(directory).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  for episode in episodes:
    identifier = str(uuid.uuid4().hex)
    length = len(episode['reward'])
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
      np.savez_compressed(f1, **episode)
      f1.seek(0)
      with filename.open('wb') as f2:
        f2.write(f1.read())

def make_env(config, datadir = '/home/rafael/IRIS/dm_control_data/hopper_hop', store = True, size = (64, 64)):
  datadir = config.dir
  suite, task = config.task.split('_', 1)
    env = wrappers.D4RL(task, config, size=size)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  callbacks = []
  if store:
    callbacks.append(lambda ep: save_episodes(datadir, [ep]))
  env = wrappers.Collect(env, callbacks, config.precision)
  #env = wrappers.GymWrapper(env)
  
  #env.metadata = {}
  #env.reward_range = (-np.inf, np.inf)
  #env.spec = None
  
  #env =  Monitor(env, datadir)
  #env = wrappers.RewardObs(env)
  #env.render()
  return env

env = make_env(config, store = True, size = (config.im_size, config.im_size))
