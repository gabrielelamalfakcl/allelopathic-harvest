#!/bin/bash

# Define the base directory where the policy folders are located
base_dir="./"

# Define the directories containing the scripts
script_dirs=(
  ./PPO_DP
  ./PPO_CSP
  ./PPO_COUNTFAIR
)

# Define CUDA device management
num_cuda_devices=4
cuda_device=0

# Define valid alpha and beta values
alpha_values=(0.5 0.75 0.9)
beta_values=(0.5 0.75 0.9)

# Iterate over all policy folders in base_dir
for folder in "$base_dir"/*; do
  # Get the folder name
  folder_name=$(basename "$folder")

  # Determine the script directory based on the folder's suffix
  # if [[ "$folder_name" == *dp ]]; then
  #   script_dir="${script_dirs[0]}"
  # if [[ "$folder_name" == *csp ]]; then
  #   script_dir="${script_dirs[1]}"
  if [[ "$folder_name" == *cf ]]; then
    script_dir="${script_dirs[2]}"
  else
    echo "Skipping folder $folder_name: does not match dp, csp, or cf suffix."
    continue
  fi


  # Extract alpha and beta values from the folder name
  alpha=$(echo "$folder_name" | grep -oP "(?<=alpha=)[^,]+")
  beta=$(echo "$folder_name" | grep -oP "(?<=beta=)[^]]+")

  # Check if the extracted alpha and beta values are in the allowed lists
  if [[ ! " ${alpha_values[@]} " =~ " ${alpha} " ]] || [[ ! " ${beta_values[@]} " =~ " ${beta} " ]]; then
    echo "Skipping folder $folder_name: alpha=$alpha, beta=$beta not in allowed values."
    continue
  fi

  # Generate a unique session name
  session_name="test_${folder_name}"

  # Command to run the test script
  cmd="python \"$script_dir/[PPO]test.py\" --alpha=$alpha --beta=$beta --cuda=$cuda_device"

  # Debugging output
  echo "Running command: $cmd"
  echo "Working directory: $folder"
  echo "Session name: $session_name"
  echo "Script directory: $script_dir"

  # Start the screen session
  screen -dmS "$session_name" bash -c "
    echo 'Starting test for folder $folder' | tee \"$folder/test.log\";
    echo 'Alpha: $alpha, Beta: $beta, CUDA: $cuda_device' | tee -a \"$folder/test.log\";
    cd \"$folder\" || { echo 'Failed to change directory to $folder'; exit 1; }
    $cmd 2>&1 | tee -a \"$folder/test.log\";
    echo 'Test completed for folder $folder' | tee -a \"$folder/test.log\";
  "

  # Update CUDA device (round-robin assignment)
  cuda_device=$(( (cuda_device + 1) % num_cuda_devices ))

  # Debugging confirmation
  echo "Launched test session for $folder_name using script $script_dir/[PPO]test.py on CUDA device $cuda_device"
  echo "Logs in $folder/test.log"
done

echo "All test sessions have been launched. Use 'screen -ls' to view active sessions."