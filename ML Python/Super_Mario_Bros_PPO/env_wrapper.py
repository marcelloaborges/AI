import numpy as np
import cv2

import gym_super_mario_bros

from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


class CustomReward(Wrapper):
    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        orig, state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]

        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return orig, state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            orig, state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return orig, self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return orig, self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        orig, state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return orig, self.states[None, :, :, :].astype(np.float32)


def process_frame(frame):
    if frame is not None:
        state = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (84, 84))[None, :, :] / 255.
        return frame, state
    else:
        return frame, np.zeros((1, 84, 84))

def process_frame_rgb(frame):
    if frame is not None:
        state = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        state = cv2.resize(state, (84, 84)) / 255.
        state = state.reshape(84, -1)
        state = state[:, :][None, :, :]
        return frame, state
    else:
        return frame, np.zeros((3, 84, 252))

def create_train_env(world, stage, actions, n_stacked_frames, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))    

    env = JoypadSpace(env, actions)
    env = CustomReward(env)
    env = CustomSkipFrame(env, skip=n_stacked_frames)

    return env