#!/bin/bash -l

#SBATCH --job-name=prompt_learn       # Job name
#SBATCH --output=prompt_learn_output_%j.out    # Output file name (%j expands to jobId)

#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks per node
#SBATCH --cpus-per-task=8        # Number of CPU cores per task
#SBATCH --mem=40G                # Memory per node
#SBATCH --time=8:00:00          # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=saml@princeton.edu
# Load necessary modules

# Activate your virtual environment
source ~/.bashrc
conda activate ltl

# Run your Python script or any other command
python soft_prompts.py

exit 0