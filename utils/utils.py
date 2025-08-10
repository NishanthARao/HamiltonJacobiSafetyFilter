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
    kaiming: bool = True,
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
        

def linear_schedule(eps_start: float, 
                    eps_end: float, 
                    duration: int,
                    t: int
                  ) -> float:
    
    slope = (eps_end - eps_start) / duration
    return max(eps_start + slope * t, eps_end)


####################### EXPERIMENTAL BASE CLASSES!! #######################

# ReplayBufferSampleExperimental = namedtuple(
#     "ReplayBufferSampleExperimental",
#     ["observations", 
#      "actions", 
#      "next_observations", 
#      "dones", 
#      "rewards"])

# class ReplayBufferExperimantal:
#     def __init__(self, 
#                  buffer_size: int, 
#                  observation_space: gym.Space,
#                  action_space: gym.Space,
#                  device: Optional[str] = "cpu",
#                  **kwargs: Dict[str, Union[int, float]],
#                 ) -> None:
        
#         self.buffer_size = buffer_size
#         self.observation_space = observation_space
#         self.action_space = action_space
#         self.device = device
        
#         self.pos = 0
#         self.full = False
        
#         action_shape = () if isinstance(action_space, gym.spaces.Discrete) else action_space.shape
#         self.observations = np.zeros((self.buffer_size, *observation_space.shape), dtype=observation_space.dtype)
#         self.actions = np.zeros((self.buffer_size, *action_shape), dtype=action_space.dtype)
#         self.next_observations = np.zeros((self.buffer_size, *observation_space.shape), dtype=observation_space.dtype)
#         self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
#         self.dones = np.zeros((self.buffer_size,), dtype=np.bool_)
        
#     def add(self, 
#             obs: np.ndarray,
#             next_obs: np.ndarray,
#             action: np.ndarray,
#             reward: float,
#             done: bool) -> None:
        
#         self.observations[self.pos] = np.array(obs).copy()
#         self.next_observations[self.pos] = np.array(next_obs).copy()
#         self.actions[self.pos] = np.array(action).copy()
#         self.rewards[self.pos] = np.array(reward).copy()
#         self.dones[self.pos] = np.array(done).copy()
        
#         self.pos += 1
#         if self.pos >= self.buffer_size:
#             self.pos = 0
#             self.full = True
            
#     def sample(self, batch_size: int) -> ReplayBufferSampleExperimental:
#         upper_bound = self.buffer_size if self.full else self.pos
#         batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
#         return ReplayBufferSampleExperimental(
#                 observations = self._to_tensor(self.observations[batch_inds]),
#                 actions = self._to_tensor(self.actions[batch_inds]),
#                 next_observations = self._to_tensor(self.next_observations[batch_inds]),
#                 dones = self._to_tensor(self.dones[batch_inds]),
#                 rewards = self._to_tensor(self.rewards[batch_inds])
#         )
        
#     def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
#         return torch.as_tensor(array, device=self.device,)


# class OffPolicyBase:
#     def __init__(self, 
#                  policy: nn.Module,
#                  env: gym.Env,
#                  learning_rate: float,
#                  buffer_size: int,
#                  learning_starts: int,
#                  batch_size: int,
#                  tau: float,
#                  gamma: float,
#                  train_freq: int,
#                  gradient_steps: int,
#                  device: str,
#                  replay_buffer_class: ReplayBufferExperimantal = ReplayBufferExperimantal,
#                 ) -> None:
        
#         self.policy = policy
#         self.env = env
#         self.learning_rate = learning_rate
#         self.num_timesteps = 0
#         self._last_obs = None
        
#         self.buffer_size = buffer_size
#         self.learning_starts = learning_starts
#         self.batch_size = batch_size
#         self.tau = tau
#         self.gamma = gamma
#         self.train_freq = train_freq
#         self.gradient_steps = gradient_steps
        
#         self.replay_buffer = replay_buffer_class(
#             buffer_size=buffer_size,
#             observation_space=env.observation_space,
#             action_space=env.action_space,
#             device="cpu"
#         )

#     def collect_rollouts(self, 
#                          num_steps: int,) -> None:
        
#         if self._last_obs is None:
#             self._last_obs, _ = self.env.reset()
            
#             for _ in range(num_steps):
#                 action, _ = self.policy.predict(self._last_obs)
                
#                 next_obs, reward, terminated, truncated, _ = self.env.step(action)
#                 done = terminated or truncated
                
#                 self.replay_buffer.add(
#                     obs=self._last_obs,
#                     next_obs=next_obs,
#                     action=action,
#                     reward=reward,
#                     done=done
#                 )
                
#                 self._last_obs = next_obs
                
#                 if done:
#                     self._last_obs, _ = self.env.reset()
                    
#                 self.num_timesteps += 1
                
#     def train(self) -> None:
#         raise NotImplementedError("This method should be implemented in subclasses.")
    
#     def learn(self,
#               total_timesteps: int,
#               log_interval: int = 100,
#               reset_num_timesteps: bool = True) -> None:
        
#         raise NotImplementedError("This method should be implemented in subclasses.")
    

# class DQNPolicy(nn.Module):
#     def __init__(self,
#                  observation_space: gym.Space,
#                  action_space: gym.Space,
#                  lr_schedule: float,
#                  net_arch: List = None,
#                  net: nn.Sequential = None,
#                  optimizer = None
#                 ) -> None:
        
#         super().__init__()
        
#         self.observation_space = observation_space
#         self.action_space = action_space
#         self.lr_schedule = lr_schedule
#         self.net = net
#         self.optimizer = optimizer
        
        
#         if net_arch is None:
#             net_arch = [64, 64]
            
#         obs_dim = observation_space.shape[0]
#         action_dim = action_space.n
        
#         layers = []
#         in_features = obs_dim
#         for out_features in net_arch:
#             layers.append(nn.Linear(in_features, out_features))
#             layers.append(nn.LeakyReLU())
#             in_features = out_features
#         layers.append(nn.Linear(in_features, action_dim))
        
#         self.net = nn.Sequential(*layers)
#         self.optimizer = optim.Adam(self.net.parameters(), lr=lr_schedule)
    
    
#     def forward(self, 
#                 obs: torch.Tensor
#                ) -> torch.Tensor:
        
#         return self.net(obs.float())
    
    
#     def predict(self,
#                 obs: Union[np.ndarray, torch.Tensor],
#                 deterministic: bool = True
#                ) -> Tuple[np.ndarray, np.ndarray]:
        
#         obs_tensor = torch.as_tensor(obs,)
#         is_batched = obs_tensor.dim() > 1
#         if not is_batched:
#             obs_tensor = obs_tensor.unsqueeze(0)
            
#         with torch.no_grad():
#             q_values = self.forward(obs_tensor)
#             if deterministic:
#                 actions = q_values.argmax(dim=1)
#             else:
#                 actions = torch.distributions.Categorical(logits=q_values).sample()
        
#         if not is_batched:
#             return actions.item(), None
#         return actions.numpy(), None
        