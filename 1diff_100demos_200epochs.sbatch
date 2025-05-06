#!/bin/bash
###############################################################################
##                                                                           ##
##                           Campus Cluster                                  ##
##                   Sample Python Job Batch Script                          ##
##                                                                           ##
###############################################################################
#                                                                            
#SBATCH --time=04:00:00                  # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Using 1 of the 3 available nodes
#SBATCH --ntasks-per-node=32             # Using all 128 cores (2x64) per node
#SBATCH --cpus-per-task=1                # 1 CPU per task
#SBATCH --mem=64G                       # Almost all of the 1TB RAM available
#SBATCH --job-name=rl_training           # Name of batch job
#SBATCH --gres=gpu:H100:1                       # Request 1 NVIDIA A10 GPU
#SBATCH --partition=secondary            # Partition (queue)
#SBATCH --account=25sp-ece598sg1-eng     # Batch account to use   
#SBATCH --output=rl_training.o%j         # Name of batch job output file
##SBATCH --mail-user=jx30@illinois.edu   # Send email notifications
##SBATCH --mail-type=BEGIN,END           # Type of email notifications to send
#                                                                            
###############################################################################


# # For Diffusion Policy batch job 1991461
python train.py --config-name=train_diffusion_unet task_name=stack_d1