#!/bin/bash

algo=${1}
env=${2:-"FetchPickAndPlace-v4"}
checkpoint_name=${3}
num_episodes=${4:-100}

python -m scripts.eval_video \
    --env_id ${env} \
    --checkpoint_name ${checkpoint_name} \
    --algo ${algo} \
    --num_episodes ${num_episodes} \


