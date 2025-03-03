export CONTROL_FREQ=50
export PORT_OFFSET=0
export USE_TORQUE_ACTIONS=False
export GAMMA=0.95

export WANDB_ENABLE_TRACKING=False
export WANDB_NAME="asdafkjahdkasj"

# Note: num_actors must be >= num_learners, but usually one actor per learner is sufficient as the bottleneck is the learner's processing power.
# Note: ntasks must be manually defined. Assuming one actor per learner, ntasks=3+num_learners (1 game_mgr, 1 actor, 1 learner, num_learners actors).
# Node: make sure to update pmc.conf to reflect the number of tasks, and the environment variable for each actor.
export NUM_LEARNERS=2

# Ensure all child processes are terminated when this script is terminated
trap 'echo "Ctrl+C pressed. Terminating all child processes..."; trap - SIGINT SIGTERM; kill -- -$$' SIGINT SIGTERM

bash example_pmc_train.sh model_pool &
bash example_pmc_train.sh league_mgr &
bash example_pmc_train.sh actor 0 &
bash example_pmc_train.sh actor 0 &
bash example_pmc_train.sh learner &

wait
