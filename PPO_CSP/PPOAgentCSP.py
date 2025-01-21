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
Update method for PPO + conditional stat parity
"""

class PPOAgentCSP(PPOAgent):
    def __init__(self, *args, alpha, beta, value_loss_coef=1.0, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.value_loss_coef = value_loss_coef
        self.config = config
       
        self.preference_index = 4  # Berry preference stored as the fourth value in the state tensor
        self.scaler = GradScaler()
        self.fairness_metrics_rewards = GetMetrics()
        self.fairness_metrics_state = FairnessMetrics()
        self.reward_normaliser_G1 = Normaliser()
        self.reward_normaliser_G2 = Normaliser()
        self.state_value_normaliser_G1 = Normaliser() 
        self.state_value_normaliser_G2 = Normaliser()
                
        self.red_non_sensitive_rewards = []  # Rewards for non_sensitive agents with red preference
        self.red_sensitive_rewards = []  # Rewards for sensitive agents with red preference
        self.blue_non_sensitive_rewards = []  # Rewards for non_sensitive agents with blue preference
        self.blue_sensitive_rewards = []  # Rewards for sensitive agents with blue preference

        
    def _update_memory(self, memory, other_agent_memory):
        cumulative_loss = 0
        num_updates = 0
        memory_states = torch.stack(memory.states).to(self.device)
        memory_actions = torch.tensor(np.array(memory.actions)).to(self.device)
        memory_rewards = torch.tensor(np.array(memory.rewards), dtype=torch.float32).to(self.device)
        memory_logprobs = torch.stack(memory.logprobs).to(self.device)
        memory_state_values = torch.stack(memory.state_values).to(self.device)
        advantages = self.compute_gae(memory_rewards.tolist(), memory_state_values.squeeze(-1).tolist())
        discounted_rewards = advantages + memory_state_values.squeeze(-1).detach()
        
        if other_agent_memory is not None:
            # Retrieve state values for the other agent group (non_sensitive vs sensitive)
            other_agent_rewards = torch.tensor(np.array(other_agent_memory.rewards), dtype=torch.float32).to(self.device)
            other_agent_state_values = torch.stack(other_agent_memory.state_values).to(self.device)
                        
            # Apply masks based on berry preference (G1 and G2) in the current batch
            G1_mask = (memory_states[:, self.preference_index] == 1) # non protected G1
            G2_mask = (memory_states[:, self.preference_index] == 2) # non protected G2
            other_agent_G1_mask = (torch.stack(other_agent_memory.states)[:, self.preference_index] == 1).to(self.device) # protected G1
            other_agent_G2_mask = (torch.stack(other_agent_memory.states)[:, self.preference_index] == 2).to(self.device) # protected G2
            
            # Within the two group (G1 and G2), we calculate the CSP s.t. we can apply a penalty based on the subgroups of agents            
            # rewards
            total_not_protected_G1_reward = memory_rewards[G1_mask].sum()
            total_protected_G1_reward = other_agent_rewards[other_agent_G1_mask].sum()
            total_not_protected_G2_reward = memory_rewards[G2_mask].sum()
            total_protected_G2_reward = other_agent_rewards[other_agent_G2_mask].sum()
            #state values
            total_not_protected_G1_state = memory_state_values[G1_mask].mean()
            total_protected_G1_state = other_agent_state_values[other_agent_G1_mask].mean()
            total_not_protected_G2_state = memory_state_values[G2_mask].mean()
            total_protected_G2_state = other_agent_state_values[other_agent_G2_mask].mean()

            # rewards -->  pass:total_rewards_not_protected_G1, total_rewards_protected_G1, total_rewards_not_protected_G2, total_rewards_protected_G2
            csp_G1_reward, csp_G2_reward, _, _ = self.fairness_metrics_rewards.conditional_statistical_parity(
                total_not_protected_G1_reward.item(), 
                total_protected_G1_reward.item(), 
                total_not_protected_G2_reward.item(), 
                total_protected_G2_reward.item()
                )
            
            # state values --> pass: state_values_not_protected_G1, state_values_protected_G1, state_values_not_protected_G2, state_values_protected_G2
            csp_G1_state, csp_G2_state, _, _ = self.fairness_metrics_state.conditional_statistical_parity(
                total_not_protected_G1_state.item(), 
                total_protected_G1_state.item(), 
                total_not_protected_G2_state.item(), 
                total_protected_G2_state.item()
                )
            
            # If csp < 0 (unprotected rewrads < protected rewards), penalty = 0
            csp_G1_reward_penalty = max(csp_G1_reward, 0)
            csp_G2_reward_penalty = max(csp_G2_reward, 0)
            csp_G1_state_penalty = max(csp_G1_state, 0)
            csp_G2_state_penalty = max(csp_G2_state, 0)
            
        # To make the two penalty components comparable, we normalise them
        # Update normalisers with episode aggregates
        self.reward_normaliser_G1.update(csp_G1_reward_penalty)
        self.reward_normaliser_G2.update(csp_G2_reward_penalty)
        self.state_value_normaliser_G1.update(csp_G1_state_penalty)
        self.state_value_normaliser_G2.update(csp_G2_state_penalty)
        
        normalized_reward_disparity_G1 = torch.tensor(self.reward_normaliser_G1.normalise(csp_G1_reward_penalty), device=self.device, dtype=torch.float32)
        normalized_reward_disparity_G2 = torch.tensor(self.reward_normaliser_G2.normalise(csp_G2_reward_penalty), device=self.device, dtype=torch.float32)
        normalized_state_value_disparity_G1 = torch.tensor(self.state_value_normaliser_G1.normalise(csp_G1_state_penalty), device=self.device, dtype=torch.float32)
        normalized_state_value_disparity_G2 = torch.tensor(self.state_value_normaliser_G2.normalise(csp_G2_state_penalty), device=self.device, dtype=torch.float32)
                         
        # Aggregate penalties with dynamic weights for state and reward penalties
        combined_csp_penalty = self.alpha * (normalized_reward_disparity_G1 + normalized_reward_disparity_G2) + self.beta * (normalized_state_value_disparity_G1 + normalized_state_value_disparity_G2)
        
        # Use total_csp_penalty in loss calculation
        for _ in range(self.k_epochs):
            # losses_policy = []
            # losses_value = []
            # losses_entropy = []
            # penalties = []
            
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
                    entropy_loss = -0.01 * dist_entropy.mean()
                    
                    # losses_policy.append(policy_loss.detach())
                    # losses_value.append(value_loss.detach())
                    # losses_entropy.append(entropy_loss.detach())
                    
                    # penalties.append(combined_csp_penalty)
                    
                    # mean_policy_loss = torch.mean(torch.stack(losses_policy)) if losses_policy else torch.tensor(0.0, device=self.device)
                    # mean_value_loss = torch.mean(torch.stack(losses_value)) if losses_value else torch.tensor(0.0, device=self.device)
                    # mean_entropy_loss = torch.mean(torch.stack(losses_entropy)) if losses_entropy else torch.tensor(0.0, device=self.device)
                    # mean_penalty = torch.mean(torch.stack(penalties))
                    
                    standard_ppo_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
                    
                    lambda_csp = (policy_loss + self.value_loss_coef * value_loss + entropy_loss) / (combined_csp_penalty + 1e-8)
                    scaled_penalty = lambda_csp * combined_csp_penalty # penalty scaled with lambda to compare with loss
                    
                    total_loss = standard_ppo_loss + scaled_penalty
                    
                    cumulative_loss += total_loss.item()
                    num_updates += 1

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        
        average_loss = cumulative_loss / num_updates if num_updates > 0 else 0
        return average_loss, standard_ppo_loss, scaled_penalty, lambda_csp 
    
    def update(self, memory_non_sensitive, memory_sensitive):
        loss_non_sensitive, ppo_loss_non_sensitive, scaled_penalty_non_sensitive, lambda_csp_non_sensitive= self._update_memory(memory_non_sensitive, memory_sensitive)
        loss_sensitive, ppo_loss_sensitive, scaled_penalty_sensitive, lambda_csp_sensitive = self._update_memory(memory_sensitive, memory_non_sensitive)

        combined_loss = loss_non_sensitive+ loss_sensitive

        return combined_loss, loss_non_sensitive, loss_sensitive, ppo_loss_non_sensitive, ppo_loss_sensitive, scaled_penalty_non_sensitive, scaled_penalty_sensitive, lambda_csp_non_sensitive, lambda_csp_sensitive

    def log_rewards_by_preference(self, rewards, states, agent_type):
        """
        Log rewards for non_sensitive and sensitive agents based on their berry preferences (1 = red, 2 = blue).
        The preference is determined by the agent's state (fourth number in the state tensor, index = 4).
        """
        for state, reward in zip(states, rewards):
            preference = state[self.preference_index]  # Extract the preference from the state (1 = red, 2 = blue)

            if agent_type == 'non_sensitive':
                if preference == 1:  # Red berry preference
                    self.red_non_sensitive_rewards.append(reward)
                elif preference == 2:  # Blue berry preference
                    self.blue_non_sensitive_rewards.append(reward)

            elif agent_type == 'sensitive':
                if preference == 1:  # Red berry preference
                    self.red_sensitive_rewards.append(reward)
                elif preference == 2:  # Blue berry preference
                    self.blue_sensitive_rewards.append(reward)
                    
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
