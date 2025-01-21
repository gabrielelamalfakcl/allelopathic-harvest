import argparse
import torch
import numpy as np
from tqdm import tqdm
import wandb
import os
import matplotlib.pyplot as plt
from PPOEnv import Environment
from PPOAgentDP import PPOAgentDP, Memory
from BerryRegrowth import LinearRegrowth
from FairnessMetrics import FairnessMetrics

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run PPO training with dynamic configuration")
parser.add_argument("--alpha", type=float, required=True, help="Alpha value for the training run")
parser.add_argument("--beta", type=float, required=True, help="Beta value for the training run")
parser.add_argument("--num_episodes", type=int, required=True, help="Number of episodes for training")
parser.add_argument("--max_timesteps", type=int, required=True, help="Maximum timesteps per episode")
parser.add_argument("--wandb_run_name", type=str, required=True, help="Run name for wandb logging")
parser.add_argument("--cuda", type=str, required=True, help="Cuda arg")

args = parser.parse_args()

config = {
    "alpha": float(args.alpha),
    "beta": float(args.beta),
    "num_episodes": int(args.num_episodes),
    "max_timesteps": int(args.max_timesteps),
    "log_interval": 1,
    "count_interval": 1,
    "regrowth_rate": 3,
    "max_lifespan": 120,
    "spont_growth_rate": 2,
    "x_dim": 25,
    "y_dim": 15,
    "num_players": 40,
    "num_bushes": 60,
    "red_player_percentage": 0.5,
    "blue_player_percentage": 0.5,
    "red_bush_percentage": 0.5,
    "blue_bush_percentage": 0.5,
    "sensitive_percentage": 0.5,
    "input_dim": 10,
    "output_dim": 8,
    "ppo_lr": 5e-4,
    "gamma": 0.99,
    "eps_clip": 0.2,
    "k_epochs": 5,
    "batch_size": 256,
    "device": torch.device(str(args.cuda) if torch.cuda.is_available() else "cpu"),
    "wandb_project": "PPO-Training",
    "wandb_run_name": args.wandb_run_name
}

# # Normalise rewards for non_sensitive and sensitive agents separately
# def normalize_rewards(rewards, running_mean, running_std, window=100):
#     """
#     Normalize rewards over a running window for stability in training.
#     """
#     if running_mean is None:
#         running_mean = np.mean(rewards)
#         running_std = np.std(rewards) if np.std(rewards) > 0 else 1
#     else:
#         running_mean = (1 - 1/window) * running_mean + (1/window) * np.mean(rewards)
#         running_std = (1 - 1/window) * running_std + (1/window) * (np.std(rewards) if np.std(rewards) > 0 else 1)

#     normalized_rewards = (rewards - running_mean) / (running_std if running_std > 0 else 1)
#     return normalized_rewards, running_mean, running_std

fairness_metrics = FairnessMetrics()

def train_ppo(ppo_agent_non_sensitive, ppo_agent_sensitive, config, render, verbose=False):
    memory_non_sensitive= Memory()
    memory_sensitive = Memory()

    cumulative_rewards_non_sensitive= []
    cumulative_rewards_sensitive = []

    ppo_loss_non_sensitive_list = []
    ppo_loss_sensitive_list = []
    penalties_non_sensitive_list = []
    lambda_dp_non_sensitive_list = []
    penalties_sensitive_list = []
    lambda_dp_sensitive_list = []
    
    rolling_window_size = 50  # Define the size of the rolling window for averaging rewards
    early_stop_patience = 100  # Number of episodes to wait before stopping if no improvement
    early_stop_threshold = 0.02  # Minimum change in rolling average to consider improvement
    # best_mean_reward = -float('inf')
    early_stop_counter = 0
    
    env = create_environment(config, verbose=verbose)

    for episode in tqdm(range(config["num_episodes"])):
        state = env.reset(env, config["regrowth_rate"], config["max_lifespan"], config["spont_growth_rate"])

        episode_reward_non_sensitive= 0
        episode_reward_sensitive = 0

        for t in range(config["max_timesteps"]):
            non_sensitive_states = [s for s in state if s[2] == 0]
            sensitive_states = [s for s in state if s[2] == 1]   
            
            non_sensitive_states_np = np.array(non_sensitive_states, dtype=np.float32)
            sensitive_states_np = np.array(sensitive_states, dtype=np.float32) 
            
            # Transform to tensors
            state_tensor_non_sensitive= torch.tensor(non_sensitive_states_np).to(ppo_agent_non_sensitive.device)
            state_tensor_sensitive = torch.tensor(sensitive_states_np).to(ppo_agent_sensitive.device)

            # Actions, log_probs and state_values
            actions_non_sensitive, log_probs_non_sensitive, state_values_non_sensitive= ppo_agent_non_sensitive.select_actions(state_tensor_non_sensitive)
            actions_sensitive, log_probs_sensitive, state_values_sensitive = ppo_agent_sensitive.select_actions(state_tensor_sensitive)
            
            actions = np.concatenate((actions_non_sensitive, actions_sensitive))
            
            # Execute actions (step)
            next_state, rewards, done = env.step(actions, env, config["regrowth_rate"], LinearRegrowth().regrowth, config["max_lifespan"], config["spont_growth_rate"])
            
            # Normalize rewards separately for non_sensitive and sensitive agents
            non_sensitive_rewards = [r for i, r in enumerate(rewards) if state[i][2] == 0]
            sensitive_rewards = [r for i, r in enumerate(rewards) if state[i][2] == 1]
            
            episode_reward_non_sensitive+= np.sum(non_sensitive_rewards)
            episode_reward_sensitive += np.sum(sensitive_rewards)

            non_sensitive_index = 0  # Counter for non_sensitive agents
            sensitive_index = 0  # Counter for sensitive agents

            for i, r in enumerate(rewards):
                if state[i][2] == 0:  # non_sensitive agent
                    r = non_sensitive_rewards[non_sensitive_index]  
                    memory_non_sensitive.states.append(state_tensor_non_sensitive[non_sensitive_index].detach())
                    memory_non_sensitive.actions.append(actions_non_sensitive[non_sensitive_index])
                    memory_non_sensitive.logprobs.append(log_probs_non_sensitive[non_sensitive_index].detach())
                    memory_non_sensitive.state_values.append(state_values_non_sensitive[non_sensitive_index].detach())
                    memory_non_sensitive.rewards.append([r])
                    non_sensitive_index += 1  # Increment the non_sensitive agent counter

                elif state[i][2] == 1:  # Sensitive agent
                    r = sensitive_rewards[sensitive_index]  
                    memory_sensitive.states.append(state_tensor_sensitive[sensitive_index].detach())
                    memory_sensitive.actions.append(actions_sensitive[sensitive_index])
                    memory_sensitive.logprobs.append(log_probs_sensitive[sensitive_index].detach())
                    memory_sensitive.state_values.append(state_values_sensitive[sensitive_index].detach())
                    memory_sensitive.rewards.append([r])
                    sensitive_index += 1  # Increment the sensitive agent counter

            state = next_state

            if done:
                break

        cumulative_rewards_non_sensitive.append(episode_reward_non_sensitive)
        cumulative_rewards_sensitive.append(episode_reward_sensitive)
        
        #combined_loss, loss_non_sensitive, loss_sensitive, ppo_loss_non_sensitive, ppo_loss_sensitive, penalty_non_sensitive, 
        # penalty_sensitive, lambda_dp_non_sensitive, lambda_dp_sensitive
 
        combined_loss, loss_non_sensitive, _, ppo_loss_non_sensitive, _, scaled_penalty_non_sensitive, _, lambda_dp_non_sensitive, _  = ppo_agent_non_sensitive.update(memory_non_sensitive, memory_sensitive)
        combined_loss, _, loss_sensitive, _, ppo_loss_sensitive, _, scaled_penalty_sensitive, _, lambda_dp_sensitive = ppo_agent_sensitive.update(memory_sensitive, memory_non_sensitive)   
    
        ppo_loss_non_sensitive_list.append(ppo_loss_non_sensitive)
        ppo_loss_sensitive_list.append(ppo_loss_sensitive)
        penalties_non_sensitive_list.append(scaled_penalty_non_sensitive.clone().detach().to(config["device"]))
        lambda_dp_non_sensitive_list.append(lambda_dp_non_sensitive.clone().detach().to(config["device"]))
        penalties_sensitive_list.append(scaled_penalty_sensitive.clone().detach().to(config["device"]))
        lambda_dp_sensitive_list.append(lambda_dp_sensitive.clone().detach().to(config["device"]))

        # Clear memories
        memory_non_sensitive.clear_memory()
        memory_sensitive.clear_memory()
        
        # Early stopping and logging based on rolling averages and DP penalties
        if len(cumulative_rewards_non_sensitive) >= rolling_window_size:
            # Compute rolling averages for rewards
            rolling_avg_non_sensitive= np.mean(cumulative_rewards_non_sensitive[-rolling_window_size:])
            rolling_avg_sensitive = np.mean(cumulative_rewards_sensitive[-rolling_window_size:])

            # Compute rolling averages for DP penalties
            rolling_avg_ppo_loss_non_sensitive= torch.mean(torch.stack(ppo_loss_non_sensitive_list[-rolling_window_size:]))
            rolling_avg_ppo_loss_sensitive = torch.mean(torch.stack(ppo_loss_sensitive_list[-rolling_window_size:]))
            rolling_avg_penalty_non_sensitive= torch.mean(torch.stack(penalties_non_sensitive_list[-rolling_window_size:]))
            rolling_avg_lambda_dp_non_sensitive= torch.mean(torch.stack(lambda_dp_non_sensitive_list[-rolling_window_size:]))
            rolling_avg_penalties_sensitive = torch.mean(torch.stack(penalties_sensitive_list[-rolling_window_size:]))
            rolling_avg_lambda_dp_sensitive = torch.mean(torch.stack(lambda_dp_sensitive_list[-rolling_window_size:]))
            
            # Log metrics
            wandb.log({
                "Rolling Average Reward (Non Sensitive)": rolling_avg_non_sensitive,
                "Rolling Average Reward (Sensitive)": rolling_avg_sensitive,
                "Rolling PPO Loss (Non Sensitive)": rolling_avg_ppo_loss_non_sensitive.item(),
                "Rolling PPO Loss (Sensitive)": rolling_avg_ppo_loss_sensitive.item(),
                "Rolling DP Penalty (Non Sensitive)": rolling_avg_penalty_non_sensitive.item(),
                "Rolling Lambda (Non Sensitive)": rolling_avg_lambda_dp_non_sensitive.item(),
                "Rolling DP Penalty (Sensitive)": rolling_avg_penalties_sensitive.item(),
                "Rolling Lambda (Sensitive)": rolling_avg_lambda_dp_sensitive.item(),
                "Total Loss (Non Sensitive)": loss_non_sensitive,
                "Total Loss (Sensitive)": loss_sensitive
            })

            # Early stopping checks based on a combination of reward improvement and DP penalty reduction
            # 1. calculate current cumulative rewards protected non-protected /2
            # 2. abs(1- (divide 1 by the prec reward))
            # 3. 2 must be < % (e.g., 2%) 
            if t > rolling_window_size:
                current_cumulative_rew = (cumulative_rewards_non_sensitive[-1] + cumulative_rewards_sensitive[-1]) / 2
                prec_cumulative_rew = (cumulative_rewards_non_sensitive[-2] + cumulative_rewards_sensitive[-2]) / 2
                value_to_check = np.abs(1 - current_cumulative_rew / prec_cumulative_rew)
                
                early_stop_counter +=1
                
                if value_to_check > early_stop_threshold:
                    early_stop_counter = 0   
                
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping triggered at episode {episode}")
                    break          
                   
            # # current_dp_penalty = (rolling_avg_penalty_non_sensitive+ rolling_avg_penalties_sensitive) / 2
            # if current_mean_reward > best_mean_reward + early_stop_threshold:
            #     best_mean_reward = current_mean_reward
            #     early_stop_counter = 0  # Reset counter if improvement found
            # # elif current_dp_penalty < best_dp_penalty - early_stop_threshold:
            # #     best_dp_penalty = current_dp_penalty
            # #     early_stop_counter = 0  # Reset counter if improvement in DP
            # else:
            #     early_stop_counter += 1  # Increment counter if no improvement

            # # Stop training if no improvement for the given patience
            # if early_stop_counter >= early_stop_patience:
            #     print(f"Early stopping triggered at episode {episode}")
            #     break

    return cumulative_rewards_non_sensitive, cumulative_rewards_sensitive


def create_environment(config, verbose=False):
    environment = Environment(x_dim=config["x_dim"], y_dim=config["y_dim"], max_steps=config["max_timesteps"],
                              num_players=config["num_players"], num_bushes=config["num_bushes"],
                              red_player_percentage=config["red_player_percentage"],
                              blue_player_percentage=config["blue_player_percentage"],
                              red_bush_percentage=config["red_bush_percentage"],
                              blue_bush_percentage=config["blue_bush_percentage"],
                              sensitive_percentage=config["sensitive_percentage"], 
                              max_lifespan=config["max_lifespan"], spont_growth_rate=config["spont_growth_rate"], 
                              verbose=verbose)
    return environment


if __name__ == '__main__':
    print(f"Using device: {config['device']}")

    ppo_agent_non_sensitive= PPOAgentDP(input_dim=config["input_dim"], output_dim=config["output_dim"], lr=config["ppo_lr"], 
                     gamma=config["gamma"], eps_clip=config["eps_clip"], k_epochs=config["k_epochs"], 
                     batch_size=config["batch_size"], device=config["device"], config=config, alpha=config["alpha"], beta=config["beta"])
    
    ppo_agent_sensitive = PPOAgentDP(input_dim=config["input_dim"], output_dim=config["output_dim"], lr=config["ppo_lr"], 
                     gamma=config["gamma"], eps_clip=config["eps_clip"], k_epochs=config["k_epochs"], 
                     batch_size=config["batch_size"], device=config["device"], config=config, alpha=config["alpha"], beta=config["beta"])

    # wandb.init(project=config["wandb_project"], name=f"[dual-policy]reward-policy-ts={config['max_timesteps']}-ne={config['num_episodes']}")
    wandb.init(
    project=config["wandb_project"],
    name=f"{config['wandb_run_name']}_alpha_{config['alpha']}_beta_{config['beta']}"
    )
    wandb.watch(ppo_agent_non_sensitive.policy_net, log="all")
    wandb.watch(ppo_agent_sensitive.policy_net, log="all")
    
    rewards = train_ppo(ppo_agent_non_sensitive, ppo_agent_sensitive, config, render=True, verbose=False)

    weights_dir = f"[alpha={config['alpha']},beta={config['beta']}]policy-weights-dp"

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Save weights for both agents
    weights_file_path_non_sensitive= os.path.join(weights_dir, '[NONSENSITIVE]ppo_agent_weights.pth')
    ppo_agent_non_sensitive.save_weights(weights_file_path_non_sensitive)

    weights_file_path_sensitive = os.path.join(weights_dir, '[SENSITIVE]ppo_agent_weights.pth')
    ppo_agent_sensitive.save_weights(weights_file_path_sensitive)
    
    wandb.finish()
    print('Training finished')