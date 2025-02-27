#!/bin/bash

#SBATCH --job-name=lifelike_agility_and_play
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --time=23:59:59

#SBATCH --gres=gpu:1

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lifelike

export CONTROL_FREQ=50
export PORT_OFFSET=0
export USE_TORQUE_ACTIONS=False
export GAMMA=0.95

export WANDB_ENABLE_TRACKING=True
export WANDB_ENTITY=keagan
export WANDB_PROJECT=lifelike_agility_and_play
export WANDB_GROUP=phase_1
export WANDB_NAME=pd_control_8192_batch_size
export WANDB_NOTES="Increase batch size to 8192"

# Note: num_actors must be >= num_learners, but usually one actor per learner is sufficient as the bottleneck is the learner's processing power.
# Note: ntasks must be manually defined. Assuming one actor per learner, ntasks=3+num_learners (1 game_mgr, 1 actor, 1 learner, num_learners actors).
# Node: make sure to update pmc.conf to reflect the number of tasks, and the environment variable for each actor.
export NUM_LEARNERS=1


# Launch all tasks concurrently using multi-program mode.
srun --multi-prog pmc.conf
