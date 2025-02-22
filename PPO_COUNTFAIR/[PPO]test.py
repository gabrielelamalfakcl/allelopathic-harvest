import argparse
import torch
from PPOEnv import Environment
from BerryRegrowth import LinearRegrowth
from GetFairness import GetMetrics
import numpy as np
import os
import pickle
import importlib.util
import sys

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
    "sensitive_percentage": 0.0,  # Adjusted for the environment creation
    "input_dim": 10,
    "output_dim": 8,
    "ppo_lr": 5e-4,
    "gamma": 0.99,
    "eps_clip": 0.2,
    "k_epochs": 5,
    "batch_size": 256,
    "device": torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"),
    "wandb_project": "PPO-Testing"
}

num2action = {0: "stay", 1: "move_up", 2: "move_down", 3: "move_left", 
              4: "move_right", 5: "ripe_eat_fruit", 6: "change_bush_color", 7: "interact_with_nearby_player"}

action_frequencies = {
    'cf_non_sensitive': {action: 0 for action in num2action.values()},
    'cf_sensitive': {action: 0 for action in num2action.values()}
}


def write_results_to_file(fairness_metrics, filename):
    """ Write fairness metrics to a file with a custom filename """
    file_path = os.path.join(os.getcwd(), filename)
    with open(file_path, 'wb') as f:
        pickle.dump(fairness_metrics, f)
    print(f"Fairness metrics successfully saved to {filename}")

# Load the trained policies for non_sensitive and sensitive agents
def load_weights(config, folder=None):
    print("Current working directory:", os.getcwd())

    # Load counterfactual fairness PPO policies for non_sensitive and sensitive agents
    ppo_agent_non_sensitive= CounterfactualPPOAgent(input_dim=config["input_dim"], output_dim=config["output_dim"], lr=config["ppo_lr"], 
                                            gamma=config["gamma"], eps_clip=config["eps_clip"], k_epochs=config["k_epochs"], 
                                            batch_size=config["batch_size"], device=config["device"], alpha=config["alpha"], beta=config["beta"])

    ppo_agent_sensitive = CounterfactualPPOAgent(input_dim=config["input_dim"], output_dim=config["output_dim"], lr=config["ppo_lr"], 
                                                gamma=config["gamma"], eps_clip=config["eps_clip"], k_epochs=config["k_epochs"], 
                                                batch_size=config["batch_size"], device=config["device"], alpha=config["alpha"], beta=config["beta"])

    weights_dir = os.path.abspath(folder)
    non_sensitive_weights_path = os.path.join(weights_dir, '[NONSENSITIVE]ppo_agent_weights.pth')
    sensitive_weights_path = os.path.join(weights_dir, '[SENSITIVE]ppo_agent_weights.pth')

    ppo_agent_non_sensitive.load_weights(non_sensitive_weights_path)
    ppo_agent_sensitive.load_weights(sensitive_weights_path)

    return ppo_agent_non_sensitive, ppo_agent_sensitive


def create_environment(config, all_agents_non_sensitive=True, random_seed=None):
    sensitive_percentage = 0.0 if all_agents_non_sensitiveelse 1.0  # Adjust sensitive attribute based on the group
    return Environment(x_dim=config["x_dim"], y_dim=config["y_dim"], max_steps=config["max_timesteps"],
                       num_players=config["num_players"], num_bushes=config["num_bushes"],
                       red_player_percentage=config["red_player_percentage"],
                       blue_player_percentage=config["blue_player_percentage"],
                       red_bush_percentage=config["red_bush_percentage"],
                       blue_bush_percentage=config["blue_bush_percentage"],
                       sensitive_percentage=sensitive_percentage, 
                       max_lifespan=config["max_lifespan"], spont_growth_rate=config["spont_growth_rate"], 
                       regrowth_rate=config["regrowth_rate"], random_seed=random_seed)

def write_results_to_file(cf_results, cf_results_norm, filename):
    results_dict = {
        "CF": cf_results,  
        "CFN": cf_results_norm, 
    }
    
    current_directory = os.getcwd()
    
    file_path = os.path.join(current_directory, filename)
    
    # Save the dictionary to a pickle (.pkl) file
    with open(file_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"Results successfully saved to {filename}")


def run_game(ppo_agent, config, random_seed, non_sensitive=True):
    print("Starting run_game...")
    # Initialize rewards based on preferences
    episode_rewards = {'red': 0, 'blue': 0}
    
    # Create the environment
    env = create_environment(config, all_agents_non_sensitive=non_sensitive, random_seed=random_seed)
    state = env.reset()

    action_frequencies = {action_name: 0 for action_name in num2action.values()}
    
    for t in range(config["max_timesteps"]):
        # print(f"Timestep {t}")  # Log the current timestep

        # Get state tensor
        state_tensor = torch.tensor(np.array(state, dtype=np.float32)).to(ppo_agent.device)

        # Select actions
        actions, _, _ = ppo_agent.select_actions(state_tensor)
        # print(f"Actions selected: {actions}")  # Log actions taken by the agent

        # Map actions to strings
        for action in actions:
            action_name = num2action.get(action, None)
            if action_name is None:
                print(f"Warning: Unmapped action {action}. Skipping.")
                continue
            action_frequencies[action_name] += 1

        # Step in the environment
        next_state, rewards, done = env.step(actions, env, config["regrowth_rate"], LinearRegrowth().regrowth, config["max_lifespan"], config["spont_growth_rate"])
        
        # Accumulate rewards
        for i, r in enumerate(rewards):
            if state[i][4] == 1:  # Red berry preference
                episode_rewards['red'] += r
            elif state[i][4] == 2:  # Blue berry preference
                episode_rewards['blue'] += r
        
        state = next_state

        if done:
            print("Environment terminated.")
            break

    print(f"Episode rewards: {episode_rewards}")
    print(f"Action frequencies: {action_frequencies}")
    return episode_rewards, action_frequencies

def test_agents(ppo_agent_non_sensitive, ppo_agent_sensitive, config):
    fairness_metrics = GetMetrics()

    # Initialize cumulative rewards dictionary
    cumulative_rewards = {
        'non_sensitive_red': [],
        'non_sensitive_blue': [],
        'sensitive_red': [],
        'sensitive_blue': []
    }
    
    # Initialize fairness metrics
    cf_results = []
    cf_results_norm = []

    # Initialize action frequencies
    action_frequencies = {
        'cf_non_sensitive': {action: 0 for action in num2action.values()},
        'cf_sensitive': {action: 0 for action in num2action.values()}
    }
    
    random_seed = 42  # Ensure both environments have the same seed

    for episode in range(config["num_episodes"]):
        print(f"Starting episode {episode + 1}/{config['num_episodes']}")  # Debugging progress

        # Run games for each agent
        rewards_cf_non_sensitive, freq_cf_non_sensitive= run_game(ppo_agent_non_sensitive, config, random_seed, non_sensitive=True)
        rewards_cf_sensitive, freq_cf_sensitive = run_game(ppo_agent_sensitive, config, random_seed, non_sensitive=False)

        # Update action frequencies
        for action, count in freq_cf_non_sensitive.items():
            action_frequencies['cf_non_sensitive'][action] += count

        for action, count in freq_cf_sensitive.items():
            action_frequencies['cf_sensitive'][action] += count

        # Update cumulative rewards
        cumulative_rewards['non_sensitive_red'].append(rewards_cf_non_sensitive['red'])
        cumulative_rewards['non_sensitive_blue'].append(rewards_cf_non_sensitive['blue'])
        cumulative_rewards['sensitive_red'].append(rewards_cf_sensitive['red'])
        cumulative_rewards['sensitive_blue'].append(rewards_cf_sensitive['blue'])

        # Compute fairness metrics for Counterfactual PPO
        dp_cf, dp_norm_cf = fairness_metrics.demographic_parity(
            total_rewards_not_protected=rewards_cf_non_sensitive['red'] + rewards_cf_non_sensitive['blue'],
            total_rewards_protected=rewards_cf_sensitive['red'] + rewards_cf_sensitive['blue'] 
        )
        cf_results.append(dp_cf)
        cf_results_norm.append(dp_norm_cf)

    # Write the final fairness results to file
    results_filename = f"fairness_results_alpha={config['alpha']}_beta={config['beta']}.pkl"
    write_results_to_file(cf_results, cf_results_norm, filename=results_filename)

    # Save cumulative rewards and action frequencies to a file
    results_data = {
        'cumulative_rewards': cumulative_rewards,
        'action_frequencies': action_frequencies
    }
    
    episode_results_filename = f"episode_results_alpha={config['alpha']}_beta={config['beta']}.pkl"
    with open(episode_results_filename, 'wb') as f:
        pickle.dump(results_data, f)

    print(f"Results saved to 'episode_results_dp.pkl'")
    return results_data

if __name__ == '__main__':
    # Load AgentPPO.py classes from classic PPO and counterfactual fairness PPO
    """
    We need to test in two separate games (2 run_games) the simulation. 
    Game 1 with agents without protected attribute.
    Game 2 with agents with protected attribute.
    """
    cf_ppo_agent_path = './PPO_COUNTFAIR/PPOAgentCOUNTF.py'

    # # Load the classic PPOAgent class from the classic PPO path
    # spec_classic = importlib.util.spec_from_file_location("ClassicPPOAgent", classic_ppo_agent_path)
    # classic_ppo_module = importlib.util.module_from_spec(spec_classic)
    # sys.modules["ClassicPPOAgent"] = classic_ppo_module
    # spec_classic.loader.exec_module(classic_ppo_module)

    # Load the counterfactual fairness PPOAgent class from the CF PPO path
    spec_cf = importlib.util.spec_from_file_location("CounterfactualPPOAgent", cf_ppo_agent_path)
    cf_ppo_module = importlib.util.module_from_spec(spec_cf)
    sys.modules["CounterfactualPPOAgent"] = cf_ppo_module
    spec_cf.loader.exec_module(cf_ppo_module)

    # Now you can use each class with its respective alias
    # ClassicPPOAgent = classic_ppo_module.PPOAgent  # For classic PPO
    CounterfactualPPOAgent = cf_ppo_module.PPOAgentCOUNTF  # For counterfactual fairness PPO
    
    folder = os.getcwd()
    # Load classic PPO agents for non_sensitive and sensitive environments
    # classic_agent_non_sensitive, classic_agent_sensitive = load_weights(config, classic=True, folder=folder)
    
    # Load counterfactual fairness PPO agents for non_sensitive and sensitive environments
    cf_agent_non_sensitive, cf_agent_sensitive = load_weights(config, folder=folder)

    # Test agents on non_sensitive and sensitive environments
    results_data = test_agents(cf_agent_non_sensitive, cf_agent_sensitive, config)
