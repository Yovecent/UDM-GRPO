#!/bin/bash
# Common part for all nodes
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=offline

MASTER_PORT=111
RANK=0
MASTER_ADDR=172.24.44.36

accelerate launch \
    --config_file accelerate_configs/4_nodes_deepspeed.yaml \
    --num_machines 4 \
    --num_processes 32 \
    --machine_rank ${RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    scripts/train.py \
    config="configs/geneval_grpo/ursa_1.7b_ibq512.yaml" \
    experiment.name="ursa_geneval" \
    experiment.output_dir="./experiments/ursa_geneval"