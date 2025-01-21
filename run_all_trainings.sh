#!/bin/bash

# Define directories where the main script is located
script_dirs=(
  ./[Dual-Policy]PPO_DP
  #./[Dual-Policy]PPO_CSP
  #./[Dual-Policy]PPO_COUNTFAIR
)

# Define arrays of alpha and beta values
alpha_values=(1) #(0.5 0.75 0.9) #(0 0.25 1)
beta_values=(1) #(0.5 0.75 0.9) #(0 0.25 1)

# Define script name and base config parameters
script_name="[PPOvsPND]train_wandb.py"
timesteps=100
episodes=10

# Create a directory to store logs if not exists
log_dir="logs"
mkdir -p "$log_dir"

# CUDA device management
num_cuda_devices=4
cuda_device=0

# Iterate over each folder
for dir in "${script_dirs[@]}"; do
  # Iterate over alpha and beta values
  for alpha in "${alpha_values[@]}"; do
    for beta in "${beta_values[@]}"; do

      # Define a session name for screen
      session_name="$(basename "$dir")_train_alpha_${alpha}_beta_${beta}"

      # Generate command to run the script with parameters
      cmd="python $dir/$script_name --alpha=$alpha --beta=$beta --num_episodes=$episodes --max_timesteps=$timesteps --wandb_run_name=$(basename $dir)_alpha_${alpha}_beta_${beta} --cuda=cuda:$cuda_device"

      # Start the screen session
      screen -dmS "$session_name" bash -c "
        echo \"Starting session $session_name\";
        echo \"Running: $cmd\";
        $cmd > $log_dir/$session_name.log 2>&1;
        echo \"Session $session_name completed. Logs saved to $log_dir/$session_name.log\";
      "

      echo "Launched session $session_name with alpha=$alpha, beta=$beta, and CUDA=$cuda_device"

      # Update CUDA device (round-robin assignment)
      cuda_device=$(( (cuda_device + 1) % num_cuda_devices ))

    done
  done
done

echo "All sessions have been launched. Use 'screen -ls' to view active sessions."
