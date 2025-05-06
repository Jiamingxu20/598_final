#!/bin/bash

# Shell script to run two Python files
# Usage: ./run_scripts.sh

echo "Starting to run Python scripts..."

# Run the first Python file
echo "Running first Python script..."
python train.py --config-name=train_diffusion_trajectory_unet_lowdim_workspace task_name=stack_d1
first_exit_code=$?

if [ $first_exit_code -ne 0 ]; then
    echo "Error: First script failed with exit code $first_exit_code"
    exit $first_exit_code
fi

echo "First script completed successfully."

# Run the second Python file
echo "Running second Python script..."
python train.py --config-name=train_diffusion_unet task_name=stack_d1
second_exit_code=$?

if [ $second_exit_code -ne 0 ]; then
    echo "Error: Second script failed with exit code $second_exit_code"
    exit $second_exit_code
fi

echo "Second script completed successfully."
echo "All scripts have been executed."

exit 0