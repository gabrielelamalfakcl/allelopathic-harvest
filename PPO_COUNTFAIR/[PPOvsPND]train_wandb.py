import argparse
import torch
import numpy as np
from tqdm import tqdm
import wandb
import os
import matplotlib.pyplot as plt
from PPOEnv import Environment
from PPOAgentCOUNTF import PPOAgentCOUNTF, Memory
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

def train_ppo_parallel(ppo_agent_non_sensitive, ppo_agent_sensitive, config, render=False, verbose=False):
    memory_non_sensitive_game1 = Memory()
    memory_sensitive_game2 = Memory()

    cumulative_rewards_game1 = []
    cumulative_rewards_game2 = []

    ppo_loss_game1_list = []
    ppo_loss_game2_list = []
    penalty_game1_list = []
    lamda_dp_game1_list = []
    penalty_game2_list = []
    lamda_dp_game2_list = []

    rolling_window_size = 50
    early_stop_patience = 100
    early_stop_threshold = 0.02
    # best_mean_reward = -float('inf')
    # best_dp_penalty = -float('inf')
    early_stop_counter = 0

    # Create two parallel environments with the same starting conditions
    random_seed = 42  # Ensure both environments have the same starting conditions
    env_game1 = create_environment(config, all_agents_non_sensitive=True, random_seed=random_seed)
    env_game2 = create_environment(config, all_agents_non_sensitive=False, random_seed=random_seed)

    for episode in tqdm(range(config["num_episodes"])):
        # Reset both environments without extra parameters
        state_game1 = env_game1.reset()
        state_game2 = env_game2.reset()

        episode_reward_game1 = 0
        episode_reward_game2 = 0

        for t in range(config["max_timesteps"]):
            # Process actions for non_sensitive agents in both environments
            state_tensor_game1 = torch.tensor(np.array(state_game1, dtype=np.float32)).to(config["device"])
            state_tensor_game2 = torch.tensor(np.array(state_game2, dtype=np.float32)).to(config["device"])

            actions_game1, log_probs_game1, state_values_game1 = ppo_agent_non_sensitive.select_actions(state_tensor_game1)
            actions_game2, log_probs_game2, state_values_game2 = ppo_agent_sensitive.select_actions(state_tensor_game2)

            # Execute actions in both environments
            next_state_game1, rewards_game1, done_game1 = env_game1.step(actions_game1, env_game1, config["regrowth_rate"], LinearRegrowth().regrowth, config["max_lifespan"], config["spont_growth_rate"])
            next_state_game2, rewards_game2, done_game2 = env_game2.step(actions_game2, env_game2, config["regrowth_rate"], LinearRegrowth().regrowth, config["max_lifespan"], config["spont_growth_rate"])

            episode_reward_game1 += np.sum(rewards_game1)
            episode_reward_game2 += np.sum(rewards_game2)

            # Store experiences in memory for both games
            memory_non_sensitive_game1.states.extend([s.detach().to(config["device"]) for s in state_tensor_game1])            
            memory_non_sensitive_game1.actions.extend(actions_game1)
            memory_non_sensitive_game1.logprobs.extend([lp.detach() for lp in log_probs_game1])
            memory_non_sensitive_game1.state_values.extend([sv.detach() for sv in state_values_game1])
            memory_non_sensitive_game1.rewards.extend(rewards_game1)

            memory_sensitive_game2.states.extend([s.detach().to(config["device"]) for s in state_tensor_game2])
            memory_sensitive_game2.actions.extend(actions_game2)
            memory_sensitive_game2.logprobs.extend([lp.detach() for lp in log_probs_game2])
            memory_sensitive_game2.state_values.extend([sv.detach() for sv in state_values_game2])
            memory_sensitive_game2.rewards.extend(rewards_game2)

            state_game1 = next_state_game1
            state_game2 = next_state_game2

            if done_game1 or done_game2:
                break

        cumulative_rewards_game1.append(episode_reward_game1)
        cumulative_rewards_game2.append(episode_reward_game2)

        # total_loss, non_sensitive_loss, sensitive_loss, scaled_penalty_non_sensitive, scaled_penalty_sensitive, lambda_dp_non_sensitive, lambda_dp_sensitive
        # Update the PPO policy with demographic parity between the two games
        # combined_loss, loss_game1, loss_game2, ppo_loss_game1, ppo_loss_game2, scaled_penalty_game1, scaled_penalty_game2, lamda_dp_game1, lamda_dp_game2
    
        combined_loss, loss_game1, _, ppo_loss_game1, _, scaled_penalty_game1, _, lamda_dp_game1, _ = ppo_agent_non_sensitive.update(memory_non_sensitive_game1, memory_sensitive_game2)
        combined_loss, _, loss_game2, _, ppo_loss_game2, _, scaled_penalty_game2, _, lamda_dp_game2 = ppo_agent_sensitive.update(memory_sensitive_game2, memory_non_sensitive_game1)
        
        ppo_loss_game1_list.append(ppo_loss_game1)
        ppo_loss_game2_list.append(ppo_loss_game2)
        penalty_game1_list.append(scaled_penalty_game1.clone().detach().to(config["device"]))
        lamda_dp_game1_list.append(lamda_dp_game1.clone().detach().to(config["device"]))
        penalty_game2_list.append(scaled_penalty_game2.clone().detach().to(config["device"]))
        lamda_dp_game2_list.append(lamda_dp_game2.clone().detach().to(config["device"]))

        # Clear memories after each episode
        memory_non_sensitive_game1.clear_memory()
        memory_sensitive_game2.clear_memory()

        # Early stopping and logging
        if len(cumulative_rewards_game1) >= rolling_window_size:
            # Compute rolling averages for rewards
            rolling_avg_game1 = np.mean(cumulative_rewards_game1[-rolling_window_size:])
            rolling_avg_game2 = np.mean(cumulative_rewards_game2[-rolling_window_size:])

            # Compute rolling averages for DP penalties
            rolling_avg_ppo_loss_game1 = torch.mean(torch.stack(ppo_loss_game1_list[-rolling_window_size:]))
            rolling_avg_ppo_loss_game2 = torch.mean(torch.stack(ppo_loss_game2_list[-rolling_window_size:]))
            rolling_avg_penalty_game1 = torch.mean(torch.stack(penalty_game1_list[-rolling_window_size:]))
            rolling_avg_lamda_dp_game1 = torch.mean(torch.stack(lamda_dp_game1_list[-rolling_window_size:]))
            rolling_avg_penalty_game2 = torch.mean(torch.stack(penalty_game2_list[-rolling_window_size:]))
            rolling_avg_lambda_dp_game2 = torch.mean(torch.stack(lamda_dp_game2_list[-rolling_window_size:]))
            
            # Log metrics
            wandb.log({
                "Rolling Average Reward (Game 1)": rolling_avg_game1,
                "Rolling Average Reward (Game 2)": rolling_avg_game2,
                "Rolling PPO Loss (Game 1)": rolling_avg_ppo_loss_game1.item(),
                "Rolling PPO Loss (Game 2)": rolling_avg_ppo_loss_game2.item(),
                "Rolling DP Penalty (Game 1)": rolling_avg_penalty_game1.item(),
                "Rolling Lambda (Game 1)": rolling_avg_lamda_dp_game1.item(),
                "Rolling DP Penalty (Game 2)": rolling_avg_penalty_game2.item(),
                "Rolling Lambda (Game 2)": rolling_avg_lambda_dp_game2.item(),
                "Total Loss (Non Sensitive)": loss_game1,
                "Total Loss (Sensitive)": loss_game2
            })
            
            
            # Early stopping checks based on a combination of reward improvement and DP penalty reduction
            # 1. calculate current cumulative rewards protected non-protected /2
            # 2. abs(1- (divide 1 by the rolling avg reward))
            # 3. 2 must be < % (e.g., 2%) 
            if t > rolling_window_size:
                current_cumulative_rew = (cumulative_rewards_game1[-1] + cumulative_rewards_game2[-1]) / 2
                prec_cumulative_rew = (cumulative_rewards_game1[-2] + cumulative_rewards_game2[-2]) / 2
                value_to_check = np.abs(1 - current_cumulative_rew / prec_cumulative_rew)
                
                early_stop_counter +=1
                
                if value_to_check > early_stop_threshold:
                    early_stop_counter = 0   
                
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping triggered at episode {episode}")
                    break    
            
            # # Early stopping checks based on a combination of reward improvement and DP penalty reduction
            # current_mean_reward = (rolling_avg_game1 + rolling_avg_game2) / 2
            # current_dp_penalty = (rolling_avg_penalty_game1 + rolling_avg_penalty_game2) / 2
            # if current_mean_reward > best_mean_reward + early_stop_threshold:
            #     best_mean_reward = current_mean_reward
            #     early_stop_counter = 0  # Reset counter if improvement found
            # # elif current_dp_penalty < best_dp_penalty - early_stop_threshold:
            # #     best_dp_penalty = current_dp_penalty  # Reset counter if improvement in DP
            # #     early_stop_counter = 0
            # else:
            #     early_stop_counter += 1  # Increment counter if no improvement

            # # Stop training if no improvement for the given patience
            # if early_stop_counter >= early_stop_patience:
            #     print(f"Early stopping triggered at episode {episode}")
            #     break

    return cumulative_rewards_game1, cumulative_rewards_game2

def create_environment(config, all_agents_non_sensitive=True, verbose=False, random_seed=None):
    sensitive_percentage = 0.0 if all_agents_non_sensitiveelse 1.0  # 0.0 means all agents are non_sensitive, 1.0 means all are sensitive

    environment = Environment(x_dim=config["x_dim"], y_dim=config["y_dim"], max_steps=config["max_timesteps"],
                              num_players=config["num_players"], num_bushes=config["num_bushes"],
                              red_player_percentage=config["red_player_percentage"],
                              blue_player_percentage=config["blue_player_percentage"],
                              red_bush_percentage=config["red_bush_percentage"],
                              blue_bush_percentage=config["blue_bush_percentage"],
                              sensitive_percentage=sensitive_percentage,
                              max_lifespan=config["max_lifespan"], spont_growth_rate=config["spont_growth_rate"],
                              regrowth_rate=config["regrowth_rate"],
                              random_seed=random_seed,
                              verbose=verbose)
    return environment

if __name__ == '__main__':
    print(f"Using device: {config['device']}")

    ppo_agent_non_sensitive= PPOAgentCOUNTF(input_dim=config["input_dim"], output_dim=config["output_dim"], lr=config["ppo_lr"],
                                gamma=config["gamma"], eps_clip=config["eps_clip"], k_epochs=config["k_epochs"],
                                batch_size=config["batch_size"], device=config["device"], config = config, alpha=config["alpha"], beta=config["beta"])

    ppo_agent_sensitive = PPOAgentCOUNTF(input_dim=config["input_dim"], output_dim=config["output_dim"], lr=config["ppo_lr"],
                                    gamma=config["gamma"], eps_clip=config["eps_clip"], k_epochs=config["k_epochs"],
                                    batch_size=config["batch_size"], device=config["device"], config = config, alpha=config["alpha"], beta=config["beta"])

    wandb.init(
    project=config["wandb_project"],
    name=f"{config['wandb_run_name']}_alpha_{config['alpha']}_beta_{config['beta']}"
    )
    wandb.watch(ppo_agent_non_sensitive.policy_net, log="all")
    wandb.watch(ppo_agent_sensitive.policy_net, log="all")

    rewards_game1, rewards_game2 = train_ppo_parallel(ppo_agent_non_sensitive, ppo_agent_sensitive, config, render=True, verbose=False)

    weights_dir = f"[alpha={config['alpha']},beta={config['beta']}]policy-weights-cf"

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Save weights for both agents
    weights_file_path_non_sensitive= os.path.join(weights_dir, '[NONSENSITIVE]ppo_agent_weights.pth')
    ppo_agent_non_sensitive.save_weights(weights_file_path_non_sensitive)

    weights_file_path_sensitive = os.path.join(weights_dir, '[SENSITIVE]ppo_agent_weights.pth')
    ppo_agent_sensitive.save_weights(weights_file_path_sensitive)

    wandb.finish()
    print('Training finished')
