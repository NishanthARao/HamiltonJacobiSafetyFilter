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
from utils.utils import ReplayBuffer, SoftQNetwork, ActorNetworkContinuous


class SafeSACContinuous:
    
    def __init__(self, 
                 env: gym.Env,
                 q_learning_rate: float = 3e-4,
                 policy_learning_rate: float = 1e-3,
                 alpha_learning_rate: Optional[float] = None,
                 buffer_size: int = int(1e6),
                 learning_starts: int = int(5e3),
                 batch_size: int = 256,
                 tau: float = 5e-3,
                 gamma: float = 0.99,
                 train_frequency: int = 1,
                 policy_update_frequency: int = 2,
                 target_network_frequency: int = 1,
                 alpha: float = 0.2,
                 autotune_alpha: bool = True,
                 wandb_log: Optional[str] = None,
                 device: str = "cpu",
                 safety_filter_args: Optional[Dict] = {},
                 dont_log: Optional[bool] = True,
                 ) -> None:
        
        self.env = env
        self.q_learning_rate = q_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.alpha_learning_rate = alpha_learning_rate if alpha_learning_rate is not None else q_learning_rate * 0.1
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_frequency = train_frequency
        self.policy_update_frequency = policy_update_frequency
        self.target_network_frequency = target_network_frequency
        self.alpha = alpha
        self.autotune_alpha = autotune_alpha
        self.wandb_log = wandb_log
        self.dont_log = dont_log
        self.device = torch.device(device)

        self.actor = ActorNetworkContinuous(self.env).to(self.device)
        self.q_network_1 = SoftQNetwork(self.env).to(self.device)
        self.q_network_2 = SoftQNetwork(self.env).to(self.device)
        self.target_q_network_1 = SoftQNetwork(self.env).to(self.device)
        self.target_q_network_2 = SoftQNetwork(self.env).to(self.device)
        self.target_q_network_1.load_state_dict(self.q_network_1.state_dict())
        self.target_q_network_2.load_state_dict(self.q_network_2.state_dict())
        
        self.q_optimizer = optim.Adam(
            list(self.q_network_1.parameters()) + list(self.q_network_2.parameters()), 
            lr=self.q_learning_rate,
            )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=self.policy_learning_rate,
            )
        
        if self.autotune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item() 
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_learning_rate)
        
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
            
            next_state_actions, next_state_log_pi, _ = self.actor.get_actions(data.next_observations)
            q1_next_target = self.target_q_network_1(data.next_observations, next_state_actions)
            q2_next_target = self.target_q_network_2(data.next_observations, next_state_actions)
            
            min_qf_next_target = (torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi)
            
            if self.enable_safety_filter:
                l_values = data.l_values.flatten()
                future_safety_val = torch.min(l_values, min_qf_next_target.view(-1))
                next_q_value = torch.where(
                                        data.dones.flatten() == 0,
                                        (1-self.gamma) * l_values + self.gamma * future_safety_val,
                                        l_values,
                                    )
                
                # Anneal gamma
                fraction = min(1.0, global_step / self.gamma_anneal_steps)
                self.gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * fraction
            else:
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(-1)
        
        q1_a_values = self.q_network_1(data.observations, data.actions).view(-1)
        q2_a_values = self.q_network_2(data.observations, data.actions).view(-1)
        q1_loss = F.mse_loss(q1_a_values, next_q_value)
        q2_loss = F.mse_loss(q2_a_values, next_q_value)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        if global_step % self.policy_update_frequency == 0:
            
            for _ in range(self.policy_update_frequency):
                
                pi, log_pi, _ = self.actor.get_actions(data.observations)

                q1_pi = self.q_network_1(data.observations, pi)
                q2_pi = self.q_network_2(data.observations, pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                    
                actor_loss = (self.alpha * log_pi - min_q_pi).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                if self.autotune_alpha:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_actions(data.observations)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                    
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
            
        if global_step % 100 == 0 and not self.dont_log:
            
            wandb.log({
                "losses/q1_mean_values": q1_a_values.mean().item(),
                "losses/q2_mean_values": q2_a_values.mean().item(),
                "losses/q1_loss": q1_loss.item(),
                "losses/q2_loss": q2_loss.item(),
                "losses/q_loss": q_loss.item() / 2.0,
                "losses/actor_loss": actor_loss.item(),
                "losses/alpha_loss": alpha_loss.item() if self.autotune_alpha else 0.0,
                "charts/alpha": self.alpha,
                "charts/SPS": int(global_step / (time.time() - self.start_time)),
                "charts/gamma": self.gamma,
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
    
    def consult_safety_filter(self,
                              observation: Union[np.ndarray, torch.Tensor],
                              task_action: Union[np.ndarray, torch.Tensor],
                              use_lrsf: bool = False,
                              use_qcbf: bool = False,
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Value here is computed from Q(s, a) as:
        # E_{a∼π(a∣s)}​[Q(s,a)−αlogπ(a∣s)]
        # =>  approx_V = min(Q1(s, a), Q2(s, a)) − alpha * log_prob
        # But if α -> 0 during training, then, approx_V = min(Q1(s, a), Q2(s, a))
        
        self.target_q_network_1.eval()
        self.target_q_network_2.eval()
        
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0) if isinstance(observation, np.ndarray) else observation
        task_action = torch.tensor(task_action, dtype=torch.float32, device=self.device).unsqueeze(0) if isinstance(task_action, np.ndarray) else task_action
        
        if task_action is None:
                return self.actor.get_actions(observation)[0], True
        
        # To estimate the safety value function
        _, _, mean_mu_s = self.actor.get_actions(observation)
        q1_da_values = self.target_q_network_1(observation, mean_mu_s)
        q2_da_values = self.target_q_network_2(observation, mean_mu_s)
        safety_val = torch.min(q1_da_values, q2_da_values).item()
        
        if use_lrsf:
            
            EPS = self.safety_filter_args.get("LRSF_SAFETY_FILTER_EPSILON", None)
            if EPS is None: raise ValueError("LRSF_SAFETY_FILTER_EPSILON must be provided if using LRSF!") 
            if safety_val > EPS:
                return task_action, False
            else:
                return self.actor.get_actions(observation)[0], True
            
        elif use_qcbf:
            
            QCBF_SAFETY_FILTER_EPSILON = self.safety_filter_args.get("QCBF_SAFETY_FILTER_EPSILON", None)
            
            if QCBF_SAFETY_FILTER_EPSILON is None: raise ValueError("QCBF_SAFETY_FILTER_EPSILON must be provided if using QCBF!")
            
            q1_ta_values = self.target_q_network_1(observation, task_action)
            q2_ta_values = self.target_q_network_2(observation, task_action)
            q_ta_value = torch.min(q1_ta_values, q2_ta_values).item()
            
            safety_threshold = QCBF_SAFETY_FILTER_EPSILON * safety_val
            
            if q_ta_value > safety_threshold:
                return task_action, False
            else:
                return self.actor.get_actions(observation)[0], True
            
        else:
            # If no safety filter is used, just return the task action
            print("\033[33mWarning: No safety filter is used. Returning task action as is.\033[0m")
            return task_action, False
            
    def learn(self,
              total_timesteps: int,
              seed: Optional[int] = None,
              ) -> None:
        
        obs, _ = self.env.reset(seed=seed)
        for global_step in track(range(total_timesteps), description="SAC Training..."):
            
            if global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _, _ = self.actor.get_actions(torch.Tensor(obs).to(self.device),)
                action = action.detach().cpu().numpy()
                
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            if info and "episode" in info and not self.dont_log:
                wandb.log({
                    "charts/episode_reward": info["episode"]["r"],
                    "charts/episode_length": info["episode"]["l"],
                    "charts/terminated_because": info["terminated_because"],
                }, step=global_step)
                
            real_next_obs = next_obs.copy()
            
            done = terminated or truncated
            self.replay_buffer.add(obs, real_next_obs, action, reward, terminated, info)
            obs = next_obs
            
            if done:
                obs, _ = self.env.reset(seed=seed)
                
            if global_step > self.learning_starts:
                self._on_step(global_step)