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
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, Union, Tuple
from utils.utils import ReplayBuffer, QNetwork, linear_schedule


class SafeDQN:
    
    def __init__(self, 
                 env: gym.Env,
                 learning_rate: float = 2.5e-4,
                 buffer_size: int = 10_000,
                 learning_starts: int = 10_000,
                 batch_size: int = 128, 
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 train_frequency: int = 10,
                 target_network_frequency: int = 500,
                 exploration_fraction: float = 0.5, 
                 exploration_start_eps: float = 1.0,
                 exploration_end_eps: float = 0.05,
                 max_grad_norm: float = 10.0,
                 wandb_log: Optional[str] = None,
                 device: str = "cpu",
                 safety_filter_args: Optional[Dict] = {},
                 ) -> None:
        
        
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_frequency = train_frequency
        self.target_network_frequency = target_network_frequency
        self.exploration_fraction = exploration_fraction
        self.exploration_start_eps = exploration_start_eps
        self.exploration_end_eps = exploration_end_eps
        self.max_grad_norm = max_grad_norm
        self.wandb_log = wandb_log
        self.device = torch.device(device)
        
        self.q_network = QNetwork(self.env).to(self.device)
        self.target_network = QNetwork(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
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
            self.gamma_anneal_steps = safety_filter_args.get("GAMMA_ANNEAL_STEPS", 1500)
        
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
            target_max, _ = self.target_network(data.next_observations).max(dim=1)
            
            if self.enable_safety_filter:
                l_values = data.l_values.flatten()
                future_safety_val = torch.min(l_values, target_max)
                td_target = (1 - self.gamma) * l_values + self.gamma * future_safety_val
                
                # Anneal gamma
                fraction = min(1.0, global_step / self.gamma_anneal_steps)
                self.gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * fraction
            else:
                td_target = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * target_max
        
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(old_val, td_target)
        
        if global_step % 100 == 0:
            wandb.log({
                "charts/gamma": self.gamma,
                "losses/td_loss": loss,
                "losses/mean_q_values": old_val.mean().item(),
                "charts/SPS": int(global_step / (time.time() - self.start_time)),
                }, step=global_step,
            )
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def _update_target(self, 
                       global_step: int, 
                      ) -> None:
        
        if global_step % self.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_network_param.data.copy_(
                    target_network_param.data * (1.0 - self.tau) + q_network_param.data * self.tau
                )
        
    def _predict_action(self, 
                        epsilon: float,
                        observation: Union[np.ndarray, torch.Tensor],
                        ) -> np.ndarray:
        
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.q_network(torch.Tensor(observation).to(self.device).unsqueeze(0))
            return q_values.argmax(dim=1).cpu().numpy().item()
    
    def _consult_safety_filter(self,
                               observation: Union[np.ndarray, torch.Tensor],
                               task_action: np.ndarray,
                               use_lrsf: bool = False,
                               use_qcbf: bool = True,
                              ) -> Tuple[np.ndarray, bool]:
        
        # Firstly, set the target network to eval mode
        self.target_network.eval()
        
        if use_lrsf:
            # Use the Least-Restrictive Safety Filter (LRSF) approach
            safety_val = self.target_network(
                torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0) if isinstance(observation, np.ndarray) else observation
            ).max(1)[0].item()
            if safety_val > self.safety_filter_args.get("SAFETY_FILTER_EPSILON", 0.5):
                return task_action, False
            else:
                safe_action = self._predict_action(epsilon=0.0, observation=observation)
                return safe_action, True
            
        elif use_qcbf:
            # Use the QCBF approach
            GAMMA_QCBF = self.safety_filter_args.get("GAMMA_QCBF", 0.99)
            q_values_obs = self.target_network(
                torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0) if isinstance(observation, np.ndarray) else observation
            )
            values_obs = q_values_obs.max(1)[0].item()
            safety_threshold = GAMMA_QCBF * values_obs
            
            safe_actions = []
            for action in range(self.env.action_space.n):
                if q_values_obs[0, action].item() >= safety_threshold:
                    safe_actions.append(action)
            
            if not safe_actions or not task_action:
                return q_values_obs.argmax(dim=1).cpu().numpy().item(), True
            else:
                if task_action in safe_actions:
                    return task_action, False
                else:
                    return safe_actions[0], True
        else:
            # If no safety filter is used, return the task action
            print("\033[33mWarning: No safety filter is used. Returning task action as is.\033[0m")
            return task_action, False

        
    
    def learn(self, 
              total_timesteps: int,
             ) -> None:
        
        obs, _ = self.env.reset()
        
        for global_step in track(range(total_timesteps), description="DQN Learning...."):
            
            epsilon = linear_schedule(
                self.exploration_start_eps,
                self.exploration_end_eps,
                self.exploration_fraction * total_timesteps,
                global_step,
            )
            
            wandb.log({
                "charts/global_step": global_step, 
                "charts/epsilon": epsilon
                }, step=global_step
            )
            
            action = self._predict_action(epsilon, obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            if info and "episode" in info:
                if info['episode']['l'] > 10:  # log only full episodes
                    wandb.log({
                            "charts/episode_length": info['episode']['l'],
                            "charts/episode_reward": info['episode']['r'],
                        }, step=global_step
                    )
                
            real_next_obs = next_obs.copy()
            # if truncated:
            #     real_next_obs = info['final_observation']
            
            done = terminated or truncated
            self.replay_buffer.add(obs, real_next_obs, action, reward, terminated, info)

            obs = next_obs
            
            if done:
                obs, _ = self.env.reset()
            
            if global_step > self.learning_starts:
                
                self._on_step(global_step)