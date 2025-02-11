#!/bin/bash

#SBATCH --job-name=lifelike_agility_and_play
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=2000
#SBATCH --time=23:59:59

#disabled --gres=gpu:1

# Launch each process in the background to run in parallel
srun --ntasks=1 bash example_epmc_train.sh model_pool &
srun --ntasks=1 bash example_epmc_train.sh league_mgr &
srun --ntasks=1 bash example_epmc_train.sh actor &
srun --ntasks=1 bash example_epmc_train.sh learner &

# Wait for all to finish
wait