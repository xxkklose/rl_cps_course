#!/bin/bash

algo=${1}
env=${2:-"FetchPickAndPlace-v4"}

SEED=42
TOTAL_TIMESTEPS=1000000
BUFFER_SIZE=1000000
LEARNING_RATE=1e-3
BATCH_SIZE=512
TAU=0.05
GAMMA=0.95
DEVICE="cuda"

case ${algo} in
    iher_sac)
        python -m scripts.train_iher_sac \
            --env ${env} \
            --seed ${SEED} \
            --total_timesteps ${TOTAL_TIMESTEPS} \
            --buffer_size ${BUFFER_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --batch_size ${BATCH_SIZE} \
            --tau ${TAU} \
            --gamma ${GAMMA} \
            --device ${DEVICE}
        ;;
    her_sac)
        python -m scripts.train_her_sac_basline \
            --env ${env} \
            --seed ${SEED} \
            --total_timesteps ${TOTAL_TIMESTEPS} \
            --buffer_size ${BUFFER_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --batch_size ${BATCH_SIZE} \
            --tau ${TAU} \
            --gamma ${GAMMA} \
            --device ${DEVICE}
        ;;
    ppo)
        # PPO 特殊参数
        python -m scripts.train_ppo_baseline \
            --env ${env} \
            --seed ${SEED} \
            --total_timesteps ${TOTAL_TIMESTEPS} \
            --learning_rate 3e-4 \
            --batch_size 64 \
            --gamma 0.99 \
            --gae_lambda 0.95 \
            --ent_coef 0.0 \
            --device ${DEVICE}
        ;;
    *)
        echo "Unknown algorithm: ${algo}"
        exit 1
        ;;
esac
