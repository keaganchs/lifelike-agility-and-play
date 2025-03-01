# Start multiple actors with different ports offsets. This allows multiple runs to be started on the same CPU to maximize utilization.
# Note: port_offset must be >= 2*num_learners (see var learner_spec in example_pmc_train.sh) if starting multiple runs on the same machine such that there is no communication interference.

# Environment 0
source pmc_config/parallel_pmc.env 0; bash example_pmc_train.sh actor 0 &

# Environment 1
source pmc_config/parallel_pmc.env 1; bash example_pmc_train.sh actor 0 &

wait