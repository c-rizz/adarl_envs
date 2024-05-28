import torch.nn as nn
from lr_gym.utils.buffers import TransitionBatch
from abc import abstractmethod

class RLPolicy(nn.Module):
    @abstractmethod
    def predict_action(self, observation, deterministic = False):
        raise NotImplementedError()
    
    @abstractmethod
    def get_hidden_state(self):
        raise NotImplementedError

    def predict(self, observation, deterministic = False):
        # Mostly for stable-baselines3 compatibility
        hidden_state = self.get_hidden_state()
        return self.predict_action(observation=observation, deterministic=deterministic), hidden_state
    
    @abstractmethod
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()
