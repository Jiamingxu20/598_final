#!/bin/bash
###############################################################################
##                                                                           ##
##                           Campus Cluster                                  ##
##                   Sample Python Job Batch Script                          ##
##                                                                           ##
###############################################################################
#                                                                            
#SBATCH --time=09:00:00                  # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Using 1 of the 3 available nodes
#SBATCH --ntasks-per-node=32             # Using all 128 cores (2x64) per node
#SBATCH --cpus-per-task=1                # 1 CPU per task
#SBATCH --mem=128G                       # Almost all of the 1TB RAM available
#SBATCH --job-name=rl_training           # Name of batch job
#SBATCH --gres=gpu:A10:1                       # Request 1 NVIDIA A10 GPU
#SBATCH --partition=eng-instruction            # Partition (queue)
#SBATCH --account=25sp-ece598sg1-eng     # Batch account to use   
#SBATCH --output=rl_training.o%j         # Name of batch job output file
##SBATCH --mail-user=jx30@illinois.edu   # Send email notifications
##SBATCH --mail-type=BEGIN,END           # Type of email notifications to send
#                                                                            
###############################################################################

# Load required modules (uncomment the one you need)
#module load python
#module load anaconda3

# Run the RL model with increased parallelism



# # For Equi Diffusion Policy batch job 1991458
# python train.py --config-name=train_equi_diffusion_unet_abs task_name=stack_d1

# # For Diffusion Policy batch job 1991461
# python train.py --config-name=train_diffusion_unet task_name=stack_d1

# For State Diffusion batch job 2068774
python train.py --config-name=train_diffusion_traj_unet_lowdim task=stack_d1
