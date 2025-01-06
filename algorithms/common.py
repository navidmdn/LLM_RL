from typing import List, Dict, Tuple
import numpy as np
import pickle
import os, pathlib


class Buffer:
    def __init__(self, buffer_size: int = 1000,):
        self.states = []
        self.actions = []
        self.rewards = []
        self.ref_actions = []
        self.buffer_size = buffer_size

    @property
    def current_size(self):
        return len(self.states)

    def add(self, states, actions, rewards, ref_actions=None):
        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)

        if ref_actions is not None:
            self.ref_actions.extend(ref_actions)

        if len(self.states) > self.buffer_size:
            self.states = self.states[-self.buffer_size:]
            self.actions = self.actions[-self.buffer_size:]
            self.rewards = self.rewards[-self.buffer_size:]
            if ref_actions is not None:
                self.ref_actions = self.ref_actions[-self.buffer_size:]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.ref_actions = []

    def sample(self, batch_size: int, ref_action: bool = False, **kwargs) -> Tuple:
        if 'sampling_type' in kwargs:
            raise NotImplementedError("Default buffer does not support sampling type. Consider passing a custom buffer to the algorithm.")

        indices = np.random.choice(len(self.states), batch_size)
        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        if ref_action:
            ref_actions = [self.ref_actions[i] for i in indices]
            return states, actions, rewards, ref_actions
        return states, actions, rewards

    def save_to_disk(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(cache_path: str):
        with open(cache_path, "rb") as f:
            buffer = pickle.load(f)
        return buffer

    def __repr__(self):
        #only printing last 10 instances
        for i in range(10):
            print(f"State: {self.states[i]}\n\n\nAction: {self.actions[i]}\n\n\nReward: {self.rewards[i]}\n\n\n")