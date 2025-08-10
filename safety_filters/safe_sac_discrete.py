import time
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
from rich.progress import track
from typing import Optional, Dict, Any, Union, Tuple
from utils.utils import ReplayBuffer, QNetwork, ActorNetworkDiscrete, linear_schedule


class SafeSACDiscrete:
    
    def __init__(self, 
                 env: gym.Env,
                 q_learning_rate: float = 3e-4,
                 policy_learning_rate: float = 3e-4,
                 buffer_size: int = int(1e6),
                 learning_starts: int = int(2e4),
                 batch_size: int = 128,
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 train_frequency: int = 4,
                 target_network_frequency: int = 500,
                 alpha: float = 0.2,
                 autotune_alpha: bool = True,
                 target_entropy_scale: float = 0.89,
                 wandb_log: Optional[str] = None,
                 device: str = "cpu",
                 safety_filter_args: Optional[Dict] = {},
                 ) -> None:
        
        self.env = env
        self.q_learning_rate = q_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_frequency = train_frequency
        self.target_network_frequency = target_network_frequency
        self.alpha = alpha
        self.autotune_alpha = autotune_alpha
        self.target_entropy_scale = target_entropy_scale
        self.wandb_log = wandb_log
        self.device = torch.device(device)

        self.actor = ActorNetworkDiscrete(self.env).to(self.device)
        self.q_network_1 = QNetwork(self.env).to(self.device)
        self.q_network_2 = QNetwork(self.env).to(self.device)
        self.target_q_network_1 = QNetwork(self.env).to(self.device)
        self.target_q_network_2 = QNetwork(self.env).to(self.device)
        self.target_q_network_1.load_state_dict(self.q_network_1.state_dict())
        self.target_q_network_2.load_state_dict(self.q_network_2.state_dict())
        
        self.q_optimizer = optim.Adam(
            list(self.q_network_1.parameters()) + list(self.q_network_2.parameters()), 
            lr=self.q_learning_rate, 
            eps=1e-4,
            )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=self.policy_learning_rate,
            eps=1e-4,
            )
        
        if self.autotune_alpha:
            self.target_entropy = -self.target_entropy_scale * torch.log(1/torch.tensor(self.env.action_space.n))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=q_learning_rate, eps=1e-4)
        
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
        )
        
        if safety_filter_args is None:
            safety_filter_args = {}
            self.enable_safety_filter = False
        else:
            self.safety_filter_args = safety_filter_args
            self.enable_safety_filter = safety_filter_args.get("USE_SAFETY_FILTER", False)
            self.gamma_start = safety_filter_args.get("GAMMA_START", 0.85)
            self.gamma_end = safety_filter_args.get("GAMMA_END", 0.999)
            self.gamma_anneal_steps = safety_filter_args.get("GAMMA_ANNEAL_STEPS", 100000)
            
        self.start_time = time.time()
            
    def _on_step(self,
                 global_step: int,
                ) -> None:
        
        if global_step % self.train_frequency == 0:
            self._train(global_step)
            
        if global_step % self.target_network_frequency == 0:
            self._update_target(global_step)
            
    def _train(self, 
               global_step: int, 
              ) -> None:
        
        data = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            
            _, next_state_log_pi, next_state_action_probs = self.actor.get_actions(data.next_observations)
            q1_next_target = self.target_q_network_1(data.next_observations)
            q2_next_target = self.target_q_network_2(data.next_observations)
            
            min_qf_next_target = (next_state_action_probs * (torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi)).sum(dim=1)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_qf_next_target
        
        q1_values = self.q_network_1(data.observations)
        q2_values = self.q_network_2(data.observations)
        q1_a_values = q1_values.gather(1, data.actions.long()).view(-1)
        q2_a_values = q2_values.gather(1, data.actions.long()).view(-1)
        q1_loss = F.mse_loss(q1_a_values, next_q_value)
        q2_loss = F.mse_loss(q2_a_values, next_q_value)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        _, log_pi, action_probs = self.actor.get_actions(data.observations)
        with torch.no_grad():
            q1_values = self.q_network_1(data.observations)
            q2_values = self.q_network_2(data.observations)
            min_q_values = torch.min(q1_values, q2_values)
            
        actor_loss = (action_probs * (self.alpha * log_pi - min_q_values)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.autotune_alpha:
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            
        if global_step % 100 == 0:
            
            wandb.log({
                "losses/q1_mean_values": q1_a_values.mean().item(),
                "losses/q2_mean_values": q2_a_values.mean().item(),
                "losses/q1_loss": q1_loss.item(),
                "losses/q2_loss": q2_loss.item(),
                "losses/actor_loss": actor_loss.item(),
                "losses/alpha_loss": alpha_loss.item() if self.autotune_alpha else 0.0,
                "charts/alpha": self.alpha,
                "charts/SPS": int(global_step / (time.time() - self.start_time)),
                }, step=global_step,
            )
            
    def _update_target(self, 
                       global_step: int, 
                      ) -> None:
        
        if global_step % self.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_q_network_1.parameters(), self.q_network_1.parameters()):
                target_network_param.data.copy_(
                    target_network_param.data * (1.0 - self.tau) + q_network_param.data * self.tau
                )
            for target_network_param, q_network_param in zip(self.target_q_network_2.parameters(), self.q_network_2.parameters()):
                target_network_param.data.copy_(
                    target_network_param.data * (1.0 - self.tau) + q_network_param.data * self.tau
                )
                
    def learn(self,
              total_timesteps: int,
              ) -> None:
        
        obs, _ = self.env.reset()
        for global_step in track(range(total_timesteps), description="SAC Training..."):
            
            if global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _, _ = self.actor.get_actions(torch.Tensor(obs).to(self.device),)
                action = action.detach().cpu().numpy()
                
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            if info and "episode" in info:
                wandb.log({
                    "charts/episode_reward": info["episode"]["r"],
                    "charts/episode_length": info["episode"]["l"],
                }, step=global_step)
                
            real_next_obs = next_obs.copy()
            
            done = terminated or truncated
            self.replay_buffer.add(obs, real_next_obs, action, reward, terminated, info)
            obs = next_obs
            
            if done:
                obs, _ = self.env.reset()
                
            if global_step >= self.learning_starts:
                self._on_step(global_step)