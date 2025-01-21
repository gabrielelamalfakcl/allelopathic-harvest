import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PPOAgent import PPOAgent
from GetFairness import GetMetrics
from FairnessMetrics import FairnessMetrics

class Normaliser:
    def __init__(self, decay=0.99):
        self.max = float('-inf')
        self.min = float('inf')
        self.decay = decay
    
    def update(self, x):
        # Apply decay to max and min
        self.max = max(x, self.max * self.decay)
        self.min = min(x, self.min * self.decay)
    
    def normalise(self, x):
        range = self.max - self.min
        if range > 0:
            return (x - self.min) / range
        return 0.0

"""
Update method for PPO + demographic parity (counterfactual between two simulations)
"""

class PPOAgentCOUNTF(PPOAgent):
    def __init__(self, *args, alpha, beta, value_loss_coef=1.0, config= None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.value_loss_coef = value_loss_coef
        self.config = config
        self.scaler = GradScaler()  # Add the scaler as an instance variable
        self.fairness_metrics_rewards = GetMetrics()
        self.fairness_metrics_state = FairnessMetrics()
        self.reward_normaliser = Normaliser()
        self.state_value_normaliser = Normaliser()
         
    def _update_memory(self, memory, other_agent_memory=None):
        cumulative_loss = 0
        num_updates = 0
        memory_states = torch.stack(memory.states).to(self.device)
        memory_actions = torch.tensor(np.array(memory.actions)).to(self.device)
        memory_rewards = torch.tensor(np.array(memory.rewards), dtype=torch.float32).to(self.device)  # Real rewards
        memory_logprobs = torch.stack(memory.logprobs).to(self.device)
        memory_state_values = torch.stack(memory.state_values).to(self.device)  # State values
        advantages = self.compute_gae(memory_rewards.tolist(), memory_state_values.squeeze(-1).tolist())
        discounted_rewards = advantages + memory_state_values.squeeze(-1).detach()

        if other_agent_memory is not None:
            other_agent_rewards = torch.tensor(np.array(other_agent_memory.rewards), dtype=torch.float32).to(self.device)
            other_agent_state_values = torch.stack(other_agent_memory.state_values).to(self.device)

            # Compute DP penalty for real rewards using GetMetrics with normalization
            dp_reward_penalty, _ = self.fairness_metrics_rewards.demographic_parity(
                total_rewards_not_protected=memory_rewards.sum().item(),
                total_rewards_protected=other_agent_rewards.sum().item()
            )

            # Compute DP penalty for state values using FairnessMetrics
            dp_state_value_penalty, _ = self.fairness_metrics_state.demographic_parity(
                state_values_not_protected=memory_state_values,
                state_values_protected=other_agent_state_values
            )
            
            # Reset penalties if negative
            dp_reward_penalty = max(dp_reward_penalty, 0)
            dp_state_value_penalty = max(dp_state_value_penalty, 0)
         
            
        # To make the two penalty components comparable, we normalise them
        # Update normalisers with episode aggregates
        self.reward_normaliser.update(dp_reward_penalty)
        self.state_value_normaliser.update(dp_state_value_penalty)

        # Normalize disparities
        normalized_reward_disparity = torch.tensor(self.reward_normaliser.normalise(dp_reward_penalty), device=self.device, dtype=torch.float32)
        normalized_state_value_disparity = torch.tensor(self.state_value_normaliser.normalise(dp_state_value_penalty), device=self.device, dtype=torch.float32)

        combined_dp_penalty = self.alpha * normalized_reward_disparity + self.beta * normalized_state_value_disparity

        for _ in range(self.k_epochs):          
            for start in range(0, len(memory_states), self.batch_size):
                end = start + self.batch_size
                batch_states = memory_states[start:end]
                batch_actions = memory_actions[start:end]
                batch_advantages = advantages[start:end].detach()
                batch_old_logprobs = memory_logprobs[start:end]
                batch_discounted_rewards = discounted_rewards[start:end]

                if batch_discounted_rewards.size(0) == 0 or batch_states.size(0) == 0:
                    continue

                batch_discounted_rewards = (batch_discounted_rewards - batch_discounted_rewards.mean()) / (batch_discounted_rewards.std() + 1e-8)

                with autocast():
                    probs = self.policy_net(batch_states)
                    dist = Categorical(probs)
                    logprobs = dist.log_prob(batch_actions)
                    dist_entropy = dist.entropy()
                    predicted_state_values = self.value_net(batch_states).squeeze()

                    ratios = torch.exp(logprobs - batch_old_logprobs)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                    mse_loss = nn.MSELoss()
                    value_loss = mse_loss(predicted_state_values, batch_discounted_rewards)
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -0.01 * dist_entropy.mean() # c2 fixed to -0.01

                    standard_ppo_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss

                    lambda_dp = (policy_loss + self.value_loss_coef * value_loss + entropy_loss) / (combined_dp_penalty + 1e-8)
                    scaled_penalty = lambda_dp * combined_dp_penalty

                    total_loss = standard_ppo_loss + scaled_penalty

                    cumulative_loss += total_loss.item()
                    num_updates += 1

                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

        average_loss = cumulative_loss / num_updates if num_updates > 0 else 0
        return average_loss, standard_ppo_loss, scaled_penalty, lambda_dp

    
    def update(self, memory_non_sensitive, memory_sensitive):
        loss_game1, ppo_loss_game1, scaled_penalty_game1, lamda_dp_game1 = self._update_memory(memory_non_sensitive, memory_sensitive)
        loss_game2, ppo_loss_game2, scaled_penalty_game2, lamda_dp_game2 = self._update_memory(memory_sensitive, memory_non_sensitive)

        combined_loss = loss_game1 + loss_game2

        return combined_loss, loss_game1, loss_game2, ppo_loss_game1, ppo_loss_game2, scaled_penalty_game1, scaled_penalty_game2, lamda_dp_game1, lamda_dp_game2
    
    def save_weights(self, file_path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict()
        }, file_path)

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        print(f"Weights loaded from {path}")

class Memory:
    def __init__(self):
        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.state_values = []
