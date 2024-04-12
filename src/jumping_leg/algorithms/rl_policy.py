import torch.nn as nn
from lr_gym.utils.buffers import TransitionBatch
from abc import abstractmethod

class RLPolicy(nn.Module):
    @abstractmethod
    def predict(self, observation, deterministic = False):
        raise NotImplementedError()

    @abstractmethod
    def update(self, transitions : TransitionBatch):
        raise NotImplementedError()
