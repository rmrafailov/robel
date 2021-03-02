#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:32:09 2021

@author: rafael
"""

import atexit
import functools
import sys
import threading
import traceback

import gym
import metaworld
import d4rl
import mujoco_py
import numpy as np
from PIL import Image

class BaodingEnv:
    
  def __init__(self, size=(128, 128)):
      from pddm.envs.baoding.baoding_env import BaodingEnv
      self._env = BaodingEnv()
      self.size = size
      
      self._env.sim_robot.renderer._camera_settings['distance'] = 0.3
      self._env.sim_robot.renderer._camera_settings['azimuth'] = -67.5
      self._env.sim_robot.renderer._camera_settings['elevation'] = -42.5


  def __getattr__(self, attr):
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)

  def step(self, action):
    state, reward, done, info = self._env.step(action)
    img = self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    obs = self._env.obs_dict.copy()
    obs['state'] = state 
    obs['image'] = img
    return obs, reward, done, info

  def reset(self):    
    state = self._env.reset()
    img = self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    obs = self._env.obs_dict.copy()
    obs['state'] = state 
    obs['image'] = img
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])





class Hammer:
    
  def __init__(self, config, size=(128, 128)):
      self._env = metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_hammer_v2.SawyerHammerEnvV2()
      self._env._last_rand_vec = np.array([-0.06, 0.4, 0.02])
      self._env._set_task_called = True
      self.size = size
      self.use_transform = config.use_transform
      self.pad = int(config.pad/2) 
      self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
      self.viewer.cam.elevation = -15
      self.viewer.cam.azimuth =  137.5
      self.viewer.cam.distance = 0.9
      self.viewer.cam.lookat[0] = -0.
      self.viewer.cam.lookat[1] = 0.6
      self.viewer.cam.lookat[2] = 0.175

      
  def __getattr__(self, attr):
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)

  def step(self, action):
    state, reward, done, info = self._env.step(action)
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    return obs, reward, done, info

  def reset(self):
    state = self._env.reset()
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
    
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    return obs

  def render(self, mode, width = 128, height = 128):
      self.viewer.render(width=width, height=width)
      img = self.viewer.read_pixels(self.size[0], self.size[1], depth=False)
      img = img[::-1]
      return img


class PegInsert:
    
  def __init__(self, size=(128, 128)):
      self._env = metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_peg_insertion_side_v2.SawyerPegInsertionSideEnvV2()
      #self._env._last_rand_vec = np.array([-0., 0.94760466, 0.1525])
      self._env._freeze_rand_vec = False
      self._env._set_task_called = True
      self.size = size
      self.use_transform = False
      self.pad = 0 
      self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)

      #self.viewer.cam.elevation = -22.5
      #self.viewer.cam.azimuth =  40.0
      #self.viewer.cam.distance = 0.825
      #self.viewer.cam.lookat[0] = -0.125
      #self.viewer.cam.lookat[1] = 0.675
      #self.viewer.cam.lookat[2] = 0.175


      self.viewer.cam.elevation = -15.0 # -22.5
      self.viewer.cam.azimuth =  130
      self.viewer.cam.distance = 0.9
      self.viewer.cam.lookat[0] = -0.125
      self.viewer.cam.lookat[1] = 0.7
      self.viewer.cam.lookat[2] = 0.1
      
  def __getattr__(self, attr):
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)

  def step(self, action):
    state, reward, done, info = self._env.step(action)
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    try:
        info['is_success'] = info['success']
    except:
        pass
    
    return obs, reward, done, info

  def reset(self):
    state = self._env.reset()
    state = self._env.reset()
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
    
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    return obs

  def render(self, mode, width = 128, height = 128):
      self.viewer.render(width=width, height=width)
      img = self.viewer.read_pixels(self.size[0], self.size[1], depth=False)
      img = img[::-1]
      return img 


class DrawerOpen:
    
  def __init__(self, config, size=(128, 128)):
      self._env = metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_v2.SawyerDrawerOpenEnvV2()
      #self._env._last_rand_vec = np.array([-0., 0.94760466, 0.1525])
      self._env._last_rand_vec = np.array([-0.1, 0.9, .0])
      self._env._set_task_called = True
      self.size = size
      self.use_transform = config.use_transform
      self.pad = int(config.pad/2) 
      self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)

      #self.viewer.cam.elevation = -22.5
      #self.viewer.cam.azimuth =  40.0
      #self.viewer.cam.distance = 0.825
      #self.viewer.cam.lookat[0] = -0.125
      #self.viewer.cam.lookat[1] = 0.675
      #self.viewer.cam.lookat[2] = 0.175


      self.viewer.cam.elevation = -22.5
      self.viewer.cam.azimuth =  15
      self.viewer.cam.distance = 0.75
      self.viewer.cam.lookat[0] = -0.15
      self.viewer.cam.lookat[1] = 0.7
      self.viewer.cam.lookat[2] = 0.10
      
  def __getattr__(self, attr):
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)

  def step(self, action):
    state, reward, done, info = self._env.step(action)
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    try:
        info['is_success'] = info['success']
    except:
        pass
    
    return obs, reward, done, info

  def reset(self):
    state = self._env.reset()
    state = self._env.reset()
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
    
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    return obs

  def render(self, mode, width = 128, height = 128):
      self.viewer.render(width=width, height=width)
      img = self.viewer.read_pixels(self.size[0], self.size[1], depth=False)
      img = img[::-1]
      return img 

class DoorOpen:
    
  def __init__(self, config, size=(128, 128)):
      self._env = metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_door_v2.SawyerDoorEnvV2()
      #self._env._last_rand_vec = np.array([.125, 1.05, .1525])
      self._env._last_rand_vec = np.array([0.0, 1.0, .1525])
      self._env._set_task_called = True
      self.size = size
      self.use_transform = config.use_transform
      self.pad = int(config.pad/2) 
      self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
#      self.viewer.cam.elevation = -10.0#-35.
#      self.viewer.cam.azimuth =  135
#      self.viewer.cam.distance = 1.1
#      self.viewer.cam.lookat[0] = 0.2
#      self.viewer.cam.lookat[1] = 0.85
#      self.viewer.cam.lookat[2] = 0.15
        
#      self.viewer.cam.elevation = -10
#      self.viewer.cam.azimuth =  117.5
#      self.viewer.cam.distance = 1.1
#      self.viewer.cam.lookat[0] = 0.125
#      self.viewer.cam.lookat[1] = 0.85
#      self.viewer.cam.lookat[2] = 0.15
      
      self.viewer.cam.elevation = -12.5
      self.viewer.cam.azimuth =  115
      self.viewer.cam.distance = 1.05
      self.viewer.cam.lookat[0] = 0.075
      self.viewer.cam.lookat[1] = 0.75
      self.viewer.cam.lookat[2] = 0.15
      
  def __getattr__(self, attr):
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)

  def step(self, action):
    state, reward, done, info = self._env.step(action)
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    try:
        info['is_success'] = info['success']
    except:
        pass
    return obs, reward, done, info

  def reset(self):
    state = self._env.reset()
    img = self.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
    
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    return obs

  def render(self, mode, width = None, height = None):
      if width is None:
          width = self.size[0]
      if height is None:
          height = self.size[1]
      self.viewer.render(width=width, height=width)
      img = self.viewer.read_pixels(width, height, depth=False)
      img = img[::-1]
      return img


class D4RL:
    
  def __init__(self, name, config, size=(64, 64), proprio = True):
      self._env = gym.make(name)
      self.name = name
      self.proprio = proprio
      if 'DClaw' in name:
          self._env.sim_scene.renderer.set_free_camera_settings(distance = 0.5,
                                                                azimuth = 180.0,
                                                                elevation = -30.0,
                                                                lookat = [0.0, 0.0, 0.2])
      self.size = size
      self.use_transform = config.use_transform
      self.pad = int(config.pad/2)
      import robel
      print(robel)
      
  def __getattr__(self, attr):
     if attr == '_wrapped_env':
       raise AttributeError()
     return getattr(self._env, attr)

  def step(self, action):
    state, reward, done, info = self._env.step(action)
    img = self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    try:
        info['is_success'] = info['score/success']
    except:
        pass
    return obs, reward, done, info

  def reset(self):
    try:
        if 'pen' in self.name:
            self._env.np_random.seed(2)
        elif 'hammer' in self.name:
            self._env.np_random.seed(32)
    except:
        pass
    
    state = self._env.reset()
    img = self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])
    if self.use_transform:
        img = img[self.pad:-self.pad, self.pad:-self.pad, :]
    obs = {'state':state, 'image':img}
    
#    if self.proprio:
#        obs_dict = self._env.get_obs_dict()
#        obs['proprio'] = obs_dict['proprio']
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])

  
class GymWrapper:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    size = 0
    for key in self._env.observation_space.spaces.keys():
        if key != 'image':
            size += np.int(np.prod(self._env.observation_space.spaces[key].shape))
    space = gym.spaces.Box(
          -np.inf, np.inf, (size,), dtype=np.float32)
    return space


  def _flatten_obs(self, obs):
        obs_pieces = []
        for key, value in obs.items():
            if key != 'image':
                flat = np.array([value]) if np.isscalar(value) else value.ravel()
                obs_pieces.append(flat)
        return np.concatenate(obs_pieces, axis=0)     

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    return self._flatten_obs(obs), reward, done, info

  def reset(self):
    obs = self._env.reset()
    return self._flatten_obs(obs)

class Collect:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    try:
        obs_dict = self.get_obs_dict()
        for key in obs_dict.keys():
            transition['obs/' + key] = obs_dict.get(key)
#        transition['proprio'] = transition['obs/proprio']
#        del transition['obs/proprio']
    except:
        pass
    
    try:
        obs_dict = self.obs_dict
        for key in obs_dict.keys():
            transition['obs/' + key] = obs_dict.get(key)
#        transition['proprio'] = transition['obs/proprio']
#        del transition['obs/proprio']
    except:
        pass
    try:
        transition['success'] = info['success']
    except:
        pass
    
    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.shape)
    transition['reward'] = 0.0
    #transition['success'] = 0.0
    transition['discount'] = 1.0
    try:
        obs_dict = self.get_obs_dict()
        for key in obs_dict.keys():
            transition['obs/' + key] = obs_dict.get(key)
#        transition['proprio'] = transition['obs/proprio']
#        del transition['obs/proprio']
    except:
        pass
    try:
        obs_dict = self.obs_dict
        for key in obs_dict.keys():
            transition['obs/' + key] = obs_dict.get(key)
#        transition['proprio'] = transition['obs/proprio']
#        del transition['obs/proprio']
    except:
        pass
    
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      print(value)
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      obs, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return obs, total_reward, done, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class ObsDict:

  def __init__(self, env, key='obs'):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self._env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = {self._key: np.array(obs)}
    return obs

class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs
