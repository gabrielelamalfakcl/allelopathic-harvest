import argparse
import torch
from PPOEnv import Environment
from PPOAgentCSP import PPOAgentCSP
from BerryRegrowth import LinearRegrowth
from GetFairness import GetMetrics
import numpy as np
import pickle
import os


parser = argparse.ArgumentParser(description="Run PPO testing with dynamic configuration")
parser.add_argument("--alpha", type=float, required=True, help="Alpha value for the test run")
parser.add_argument("--beta", type=float, required=True, help="Beta value for the test run")
parser.add_argument("--cuda", type=str, required=True, help="CUDA device to use")

args = parser.parse_args()

# Test configuration
config = {
    "alpha": args.alpha,
    "beta": args.beta,
    "num_episodes": 1000,
    "max_timesteps": 3000,
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
    "device": torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"),
    "wandb_project": "PPO-Training"
}

# Load the weights of the trained policies
def load_weights(config, folder):
    print("Current working directory:", os.getcwd())
    
    ppo_agent_non_sensitive= PPOAgentCSP(input_dim=config["input_dim"], output_dim=config["output_dim"], lr=config["ppo_lr"], 
                     gamma=config["gamma"], eps_clip=config["eps_clip"], k_epochs=config["k_epochs"], 
                     batch_size=config["batch_size"], device=config["device"], alpha=config["alpha"], beta=config["beta"])
    
    ppo_agent_sensitive = PPOAgentCSP(input_dim=config["input_dim"], output_dim=config["output_dim"], lr=config["ppo_lr"], 
                     gamma=config["gamma"], eps_clip=config["eps_clip"], k_epochs=config["k_epochs"], 
                     batch_size=config["batch_size"], device=config["device"], alpha=config["alpha"], beta=config["beta"])
    
    weights_dir = os.path.abspath(folder)
    
    non_sensitive_weights_path = os.path.join(weights_dir, '[NONSENSITIVE]ppo_agent_weights.pth')
    sensitive_weights_path = os.path.join(weights_dir, '[SENSITIVE]ppo_agent_weights.pth')

    ppo_agent_non_sensitive.load_weights(non_sensitive_weights_path)
    ppo_agent_sensitive.load_weights(sensitive_weights_path)   
     
    return ppo_agent_non_sensitive, ppo_agent_sensitive

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

def write_results_to_file(dp_results, dp_results_norm, csp_results, csp_results_norm, filename):
    """
    Write both DP and CSP results to a file.
    
    Parameters:
    - dp_results: Dictionary containing demographic parity results.
    - dp_results_norm: Dictionary containing normalized demographic parity results.
    - csp_results: Dictionary containing conditional statistical parity results.
    - csp_results_norm: Dictionary containing normalized conditional statistical parity results.
    - filename: Name of the file to save the results to.
    """
    results_dict = {
        "DP": dp_results,  # Demographic Parity results
        "DPN": dp_results_norm,  # Normalized Demographic Parity results
        "CSP": csp_results,  # Conditional Statistical Parity results
        "CSPN": csp_results_norm  # Normalized Conditional Statistical Parity results
    }

    # Get the current script's directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full file path
    file_path = os.path.join(current_directory, filename)
    
    # Save the dictionary to a pickle (.pkl) file
    with open(file_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"Results successfully saved to {file_path}")


def test_agents(ppo_agent_non_sensitive, ppo_agent_sensitive, config):
    fairness_metrics = GetMetrics()
    env = create_environment(config)
    
    # Initialize cumulative rewards dictionary
    cumulative_rewards = {
        'non_sensitive_red': [],
        'non_sensitive_blue': [],
        'sensitive_red': [],
        'sensitive_blue': []
    }
    
    # Initialize action frequency storage
    num2action = {0: "stay", 1: "move_up", 2: "move_down", 3: "move_left", 
                  4: "move_right", 5: "ripe_eat_fruit", 6: "change_bush_color", 7: "interact_with_nearby_player"}
    # Initialize action frequency storage with descriptive labels
    action_frequencies = {
        'non_sensitive': {action: 0 for action in num2action.values()},
        'sensitive': {action: 0 for action in num2action.values()}
    }
    # action_frequencies = {
    #     'non_sensitive': np.zeros(config['output_dim']),  # Assuming output_dim corresponds to number of actions
    #     'sensitive': np.zeros(config['output_dim'])
    # }
    
    dp_results = []
    dp_results_norm = []
    csp_results = []
    csp_results_norm = []

    for episode in range(config["num_episodes"]):
        state = env.reset(env, config["regrowth_rate"], config["max_lifespan"], config["spont_growth_rate"])
        
        # Initialize cumulative rewards for this episode
        episode_rewards = {
            'non_sensitive_red': 0,
            'non_sensitive_blue': 0,
            'sensitive_red': 0,
            'sensitive_blue': 0
        }

        # Loop through the timesteps for this episode
        for t in range(config["max_timesteps"]):
            non_sensitive_states = [s for s in state if s[2] == 0]
            sensitive_states = [s for s in state if s[2] == 1]   
            
            non_sensitive_states_np = np.array(non_sensitive_states, dtype=np.float32)
            sensitive_states_np = np.array(sensitive_states, dtype=np.float32) 
            
            state_tensor_non_sensitive= torch.tensor(non_sensitive_states_np).to(ppo_agent_non_sensitive.device)
            state_tensor_sensitive = torch.tensor(sensitive_states_np).to(ppo_agent_sensitive.device)

            actions_non_sensitive, _, _ = ppo_agent_non_sensitive.select_actions(state_tensor_non_sensitive)
            actions_sensitive, _, _ = ppo_agent_sensitive.select_actions(state_tensor_sensitive)
            
            actions = np.concatenate((actions_non_sensitive, actions_sensitive))
            
            # Count the action frequencies
            for action in actions_non_sensitive:
                action_label = num2action[action]  # Convert numeric action to descriptive label
                action_frequencies['non_sensitive'][action_label] += 1
            for action in actions_sensitive:
                action_label = num2action[action]  # Convert numeric action to descriptive label
                action_frequencies['sensitive'][action_label] += 1

            next_state, rewards, done = env.step(actions, env, config["regrowth_rate"], LinearRegrowth().regrowth, config["max_lifespan"], config["spont_growth_rate"])
            
            # Accumulate rewards based on sensitive attribute status and berry preference (preference is state[4])
            for i, r in enumerate(rewards):
                if state[i][2] == 0:  # non_sensitive agent
                    if state[i][4] == 1:  # Red berry preference
                        episode_rewards['non_sensitive_red'] += r
                    elif state[i][4] == 2:  # Blue berry preference
                        episode_rewards['non_sensitive_blue'] += r
                elif state[i][2] == 1:  # Sensitive agent
                    if state[i][4] == 1:  # Red berry preference
                        episode_rewards['sensitive_red'] += r
                    elif state[i][4] == 2:  # Blue berry preference
                        episode_rewards['sensitive_blue'] += r
            
            state = next_state
            
            if done:
                break
        
        # Append cumulative rewards for each episode
        cumulative_rewards['non_sensitive_red'].append(episode_rewards['non_sensitive_red'])
        cumulative_rewards['non_sensitive_blue'].append(episode_rewards['non_sensitive_blue'])
        cumulative_rewards['sensitive_red'].append(episode_rewards['sensitive_red'])
        cumulative_rewards['sensitive_blue'].append(episode_rewards['sensitive_blue'])
        
        # Calculate Demographic Parity (DP)
        dp, dp_norm = fairness_metrics.demographic_parity(
            total_rewards_not_protected=episode_rewards['non_sensitive_red'] + episode_rewards['non_sensitive_blue'],
            total_rewards_protected=episode_rewards['sensitive_red'] + episode_rewards['sensitive_blue']
        )
        dp_results.append(dp)
        dp_results_norm.append(dp_norm)
        
        # Calculate Conditional Statistical Parity (CSP)
        csp_G1, csp_G2, csp_G1_norm, csp_G2_norm = fairness_metrics.conditional_statistical_parity(
            total_rewards_not_protected_G1=episode_rewards['non_sensitive_red'],
            total_rewards_protected_G1=episode_rewards['sensitive_red'],
            total_rewards_not_protected_G2=episode_rewards['non_sensitive_blue'],
            total_rewards_protected_G2=episode_rewards['sensitive_blue']
        )
        csp_results.append({
            'CSP_G1': csp_G1,
            'CSP_G2': csp_G2
        })
        csp_results_norm.append({
            'CSP_G1': csp_G1_norm,
            'CSP_G2': csp_G2_norm
        })  
    
    # Write the final fairness results to file
    # Write the final fairness results to file
    results_filename = f"fairness_results_alpha={config['alpha']}_beta={config['beta']}.pkl"
    write_results_to_file(dp_results, dp_results_norm, csp_results, csp_results_norm, filename=results_filename)

    
    # Save cumulative rewards and action frequencies to a file
    results_data = {
        'cumulative_rewards': cumulative_rewards,
        'action_frequencies': action_frequencies
    }
    
    episode_results_filename = f"episode_results_alpha={config['alpha']}_beta={config['beta']}.pkl"
    with open(episode_results_filename, 'wb') as f:
        pickle.dump(results_data, f)
    
    print("Cumulative rewards and action frequencies saved to 'episode_results_csp.pkl'")

    return dp_results, dp_results_norm, csp_results, csp_results_norm

if __name__ == '__main__':
    folder = os.getcwd()
    
    # Load configuration
    ppo_agent_non_sensitive, ppo_agent_sensitive = load_weights(config, folder)

    # Test the loaded agents
    test_agents(ppo_agent_non_sensitive, ppo_agent_sensitive, config)