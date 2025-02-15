# Usage:
#   1. open four terminals;
#   2. "bash example_pmc_train.sh model_pool" in terminal 1;
#   3. "bash example_pmc_train.sh league_mgr" in terminal 2;
#   4. "bash example_pmc_train.sh actor" in terminal 3;
#   5. "bash example_pmc_train.sh learner" in terminal 4;


role=$1

# If role is actor, accept next arg for learner_id to communicate with
if [ $role == actor ]
then
  if [ -z "$2" ]
  then
    echo "Error: actor role requires learner_id as second argument"
    exit 1
  fi
  learner_id=$2
fi


# Get frequently changed args as environment variables
# Shorthand: foo="${ENV_VAR:-default_value}"
control_freq="${CONTROL_FREQ:-50.0}" # Default 50.0
port_offset="${PORT_OFFSET:-0}" # Defalt 0. Must be >= 2*num_learners (see var learner_spec) if starting multiple runs on the same machine such that there is no communication interference
use_torque_actions="${USE_TORQUE_ACTIONS:-False}" # Default False
gamma="${GAMMA:-0.95}" # Default 0.95

# Weights and Biases args
wandb_enable_tracking="${WANDB_ENABLE_TRACKING:-False}"
wandb_entity="${WANDB_ENTITY:-""}"
wandb_project="${WANDB_PROJECT:-""}"
wandb_group="${WANDB_GROUP:-""}"
wandb_name="${WANDB_NAME:-""}"
wandb_notes="${WANDB_NOTES:-""}"

# Set up multiple learners
num_learners="${NUM_LEARNERS:-1}"
learner_spec=""
for ((i=0; i<num_learners; i++)); do
  port1=$((30003 + port_offset + 2*i))
  port2=$((30004 + port_offset + 2*i))
  if [ $i -eq 0 ]; then
    learner_spec="0:${port1}:${port2}"
  else
    learner_spec+=",0:${port1}:${port2}"
  fi
done

# Actors will only communicate with one learner. Several actors must be started as individual jobs with the env var for actor_learner_num set to the same value as the learner's learner_id.
# Note: num_actors must be >= num_learners, but usually one actor per learner is sufficient as the bottleneck is the learner's processing power.
# Example: with two learners (learner_num=2), one learner process is started, and two actor processes must be started with the learner_id arg set to 0 and 1.
actor_learner_ports="$((30003 + port_offset + 2*learner_id)):$((30004 + port_offset + 2*learner_id))"

# common args
actor_type=PPO
outer_env=lifelike.sim_envs.pybullet_envs.create_tracking_game
outer_env_2=lifelike.sim_envs.pybullet_envs.create_tracking_env

game_mgr_type=tleague.game_mgr.game_mgrs.SelfPlayGameMgr && \
game_mgr_config="{
  'max_n_players': 1}"
mutable_hyperparam_type=ConstantHyperparam
hyperparam_config_name="{ \
  'learning_rate': 0.00001, \
  'lam': 0.95, \
  'gamma': ${gamma}, \
}" && \
policy=lifelike.networks.legged_robot.pmc_net.pmc_net
learner_policy_config="{ \
  'test': False, \
  'rl': True, \
  'use_loss_type': 'rl', \
  'z_prior_type': 'VQ', \
  'use_value_head': True, \
  'rms_momentum': 0.0001, \
  'append_hist_a': True, \
  'main_activation_func': 'relu', \
  'n_v': 1, \
  'use_lstm': False, \
  'z_len': 32, \
  'num_embeddings': 256, \
  'conditional': True, \
  'bot_neck_z_embed_size': 32, \
  'bot_neck_prop_embed_size': 64, \
}" && \
actor_policy_config="{ \
  'batch_size': 1, \
  'rollout_len': 1, \
  'test': True, \
  'use_loss_type': 'none', \
  'z_prior_type': 'VQ', \
  'use_value_head': True, \
  'rms_momentum': 0.0001, \
  'append_hist_a': True, \
  'main_activation_func': 'relu', \
  'n_v': 1, \
  'use_lstm': False, \
  'z_len': 32, \
  'conditional': True, \
  'bot_neck_z_embed_size': 32, \
  'bot_neck_prop_embed_size': 64, \
  'sync_statistics': 'none', \
}" && \
learner_config="{ \
  'vf_coef': 1, \
  'max_grad_norm': 0.5, \
  'distill_coef': 0.0, \
  'ent_coef': 0.00000, \
  'ep_loss_coef': {'q_latent_loss':1.0, 'e_latent_loss':0.25, 'rms_loss': 1.0}, \
}" && \
env_config="{ \
  'arena_id': 'LeggedRobotTracking', \
  'render': False, \
  'data_path': '../data/mocap_data', \
  'control_freq': ${control_freq}, \
  'prop_type': ['joint_pos', 'joint_vel', 'root_ang_vel_loc', 'root_lin_vel_loc', 'e_g'], \
  'prioritized_sample_factor': 3.0, \
  'set_obstacle': True, \
  'kp': 50.0, \
  'kd': 0.5, \
  'max_tau': 18, \
  'reward_weights': {'joint_pos': 0.3, 'joint_vel': 0.05, 'end_effector': 0.1, 'root_pose': 0.5, 'root_vel': 0.05,}, \
  'use_torque_actions': ${use_torque_actions}, \
}" && \

echo "Running as ${role}"

if [ $role == model_pool ]
then
# model pool
python -i -m tleague.bin.run_model_pool \
  --ports $((10003 + port_offset)):$((10004 + port_offset)) \
  --verbose 0
fi

# league mgr
if [ $role == league_mgr ]
then
python -i -m tleague.bin.run_league_mgr \
  --port=$((20005 + port_offset)) \
  --model_pool_addrs=localhost:$((10003 + port_offset)):$((10004 + port_offset)) \
  --game_mgr_type="${game_mgr_type}" \
  --game_mgr_config="${game_mgr_config}" \
  --mutable_hyperparam_type="${mutable_hyperparam_type}" \
  --hyperparam_config_name="${hyperparam_config_name}" \
  --restore_checkpoint_dir="" \
  --init_model_paths="[]" \
  --save_checkpoint_root=./tmp-trvd-yymmdd_chkpoints \
  --save_interval_secs=85 \
  --mute_actor_msg \
  --pseudo_learner_num=-1 \
  --verbose=0
fi

# learner
if [ $role == learner ]
then
python -i -m lifelike.bin.run_pg_learner \
  --learner_spec="${learner_spec}" \
  --model_pool_addrs=localhost:$((10003 + port_offset)):$((10004 + port_offset)) \
  --league_mgr_addr=localhost:$((20005 + port_offset)) \
  --learner_id=lrngrp0 \
  --unroll_length=128 \
  --rollout_length=8 \
  --batch_size=256 \
  --rm_size=1024 \
  --pub_interval=5 \
  --log_interval=4000 \
  --total_timesteps=20000000000000 \
  --burn_in_timesteps=12 \
  --outer_env="${outer_env_2}" \
  --env_config="${env_config}" \
  --policy="${policy}" \
  --policy_config="${learner_policy_config}" \
  --batch_worker_num=16 \
  --norwd_shape \
  --learner_config="${learner_config}" \
  --type=PPO \
  --track_wandb="${wandb_enable_tracking}" \
  --wandb_entity="${wandb_entity}" \
  --wandb_project="${wandb_project}" \
  --wandb_group="${wandb_group}" \
  --wandb_name="${wandb_name}" \
  --wandb_notes="${wandb_notes}"
fi

#--env="${env}" \

# actor
if [ $role == actor ]
then
python -i -m lifelike.bin.run_pg_actor \
  --model_pool_addrs=localhost:$((10003 + port_offset)):$((10004 + port_offset)) \
  --league_mgr_addr=localhost:$((20005 + port_offset)) \
  --learner_addr=localhost:"${actor_learner_ports}" \
  --unroll_length=128 \
  --update_model_freq=320 \
  --outer_env="${outer_env}" \
  --env_config="${env_config}" \
  --policy="${policy}" \
  --policy_config="${actor_policy_config}" \
  --log_interval_steps=300 \
  --n_v=1 \
  --rwd_shape \
  --nodistillation \
  --verbose=0 \
  --type="${actor_type}"
fi
