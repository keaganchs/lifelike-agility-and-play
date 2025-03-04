export CONTROL_FREQ=50
export PORT_OFFSET=5
export USE_TORQUE_ACTIONS=False
export GAMMA=0.95

export WANDB_ENABLE_TRACKING=False

# Note: num_actors must be >= num_learners, but usually one actor per learner is sufficient as the bottleneck is the learner's processing power.
# Note: ntasks must be manually defined. Assuming one actor per learner, ntasks=3+num_learners (1 game_mgr, 1 actor, 1 learner, num_learners actors).
# Node: make sure to update pmc.conf to reflect the number of tasks, and the environment variable for each actor.
export NUM_LEARNERS=2


# Map 1-4 to the respective roles for convenience, while still allowing names
case $1 in
  1|model_pool)
    role=model_pool
    ;;
  2|league_mgr)
    role=league_mgr
    ;;
  3|actor)
    role=actor
    ;;
  4|learner)
    role=learner
    ;;
  *)
  echo "Error: valid role must be specified as first argument.
Options are:
    1. model_pool
    2. league_mgr
    3. actor
    4. learner"
  exit 1
esac


# Ensure all child processes are terminated when this script is terminated
trap 'echo "Ctrl+C pressed. Terminating all child processes..."; trap - SIGINT SIGTERM; kill -- -$$' SIGINT SIGTERM

bash example_pmc_train.sh $role $2 &

wait