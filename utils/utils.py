import torch
import wandb
import psutil
import random
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions.categorical import Categorical
from typing import Union, Tuple, Dict, Optional, List, Any, NamedTuple

class ReplayBufferSample(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    l_values: torch.Tensor

class ReplayBuffer:
    
    def __init__(self, 
                 buffer_size: int, 
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 device: torch.device | str = "cpu",
                ) -> None:
        
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0] if isinstance(action_space, gym.spaces.Box) else 1
        self.pos = 0
        self.full = False
        
        self.buffer_size = buffer_size
        
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size,), dtype=np.float32)
        self.l_values = np.zeros((self.buffer_size,), dtype=np.float32)
        
    def add(self, 
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: bool,
            info: Dict[str, Any],
           ) -> None:
        
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        
        self.timeouts[self.pos] = np.array(info.get("TimeLimit.truncated", False)).copy()
        
        self.l_values[self.pos] = np.array(info.get("l_value", 0.0)).copy()
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.pos = 0
            self.full = True
            
    def sample(self,
               batch_size: int,
              ) -> ReplayBufferSample:
        
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
        return self._get_samples(batch_inds)
    
    def _get_samples(self,
                     batch_inds: np.ndarray,
                    ) -> ReplayBufferSample:
        
        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            self.next_observations[batch_inds, :],
            (self.dones[batch_inds] * (1 - self.timeouts[batch_inds])).reshape(-1, 1),
            self.rewards[batch_inds].reshape(-1, 1),
            self.l_values[batch_inds].reshape(-1, 1),
        )
        
        return ReplayBufferSample(*tuple(map(self._to_torch, data)))
        
    
    def _to_torch(self, 
                  array: np.ndarray
                ) -> torch.Tensor:
        
        return torch.tensor(array, device=self.device, )
    

def _layer_init(
    layer: nn.Module,
    bias_constant: float = 0.0,
    kaiming: bool = False,
) -> nn.Module:
    
    if kaiming:
        nn.init.kaiming_uniform_(layer.weight,)
        torch.nn.init.constant_(layer.bias, bias_constant)
        
    return layer
    
class QNetwork(nn.Module):
    
    def __init__(self,
                 env: gym.Env,
                 hidden_dim: int = 256,
                 kaiming: bool = True,
                 activation_fn: nn.Module = nn.LeakyReLU,
                ) -> None:
        
        super().__init__()

        self.network = nn.Sequential(
            _layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), hidden_dim), kaiming=kaiming),
            activation_fn(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim), kaiming=kaiming),
            activation_fn(),
            _layer_init(nn.Linear(hidden_dim, env.action_space.n), kaiming=kaiming),
        )
        
    def forward(self, 
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        return self.network(x.float())


class ActorNetworkDiscrete(nn.Module):

    def __init__(self, 
               env: gym.Env,
               hidden_dim: int = 256,
               kaiming: bool = True,
               activation_fn: nn.Module = nn.LeakyReLU,
              ) -> None:
        
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), hidden_dim), kaiming=kaiming),
            activation_fn(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim), kaiming),
            activation_fn(),
            _layer_init(nn.Linear(hidden_dim, 512), kaiming=kaiming),
            activation_fn(),
            _layer_init(nn.Linear(512, env.action_space.n), kaiming=kaiming))
        
    def forward(self, 
                x: torch.Tensor,
               ) -> torch.Tensor:
        
        # Logits for the action probabilities
        return self.network(x.float())
    
    def get_actions(self, 
                    x: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        
        action_probs = policy_dist.probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        return action, log_probs, action_probs
    
            
class SoftQNetwork(nn.Module): 

    def __init__(self, 
                 env: gym.Env,
                 hidden_dim: int = 256,
                 kaiming: bool = False,
                 activation_fn: nn.Module = nn.LeakyReLU,
                ) -> None:
        
        super().__init__()

        self.network = nn.Sequential(
            _layer_init(nn.Linear((np.array((env.observation_space.shape)).prod() + env.action_space.shape).item(), hidden_dim), kaiming=kaiming),
            activation_fn(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim), kaiming=kaiming),
            activation_fn(),
            _layer_init(nn.Linear(hidden_dim, 1), kaiming=kaiming),
        )
        
    def forward(self, 
                x: torch.Tensor,
                action: torch.Tensor,
               ) -> torch.Tensor:
        
        x = torch.cat([x, action], dim=1)
        return self.network(x.float())


class ActorNetworkContinuous(nn.Module):
    
    def __init__(self, 
                 env: gym.Env,
                 hidden_dim: int = 256,
                 kaiming: bool = False,
                 activation_fn: nn.Module = nn.LeakyReLU,
                ) -> None:
        
        super().__init__()

        self.fc1 = _layer_init(nn.Linear(np.array(env.observation_space.shape).prod().item(), hidden_dim), kaiming=kaiming)
        self.fc2 = _layer_init(nn.Linear(hidden_dim, hidden_dim), kaiming=kaiming)
        self.fc_mean = _layer_init(nn.Linear(hidden_dim, np.prod(env.action_space.shape)), kaiming=kaiming)
        self.fc_log_std = _layer_init(nn.Linear(hidden_dim, np.prod(env.action_space.shape)), kaiming=kaiming)
        self.activation_fn = activation_fn()
        
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,)
        )
        
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            )
        )
        
        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX = 2.0
        
    def forward(self, 
                x: torch.Tensor,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = self.activation_fn(self.fc1(x.float()))
        x = self.activation_fn(self.fc2(x))
        
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_log_std(x))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1.0)
        
        return mean, log_std
    
    def get_actions(self, 
                    x: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        EPS = 1e-6
        
        mean, log_std = self(x)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
    
    
def linear_schedule(eps_start: float, 
                    eps_end: float, 
                    duration: int,
                    t: int
                  ) -> float:
    
    slope = (eps_end - eps_start) / duration
    return max(eps_start + slope * t, eps_end)