#!/bin/bash
# 使用 CPU 进行多卡通信模拟，用于 debug 学习 FSDP2
# 用法:
#   bash run.sh              # 默认模拟 2 个进程
#   NUM_PROCS=4 bash run.sh  # 模拟 4 个进程
#   NUM_PROCS=1 bash run.sh  # 单进程 FSDP2 调试

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 配置：模拟的"卡数"（实际为 CPU 进程数）
NUM_PROCS=${NUM_PROCS:-2}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NUM_PROCS*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NUM_PROCS \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

logfile=NUM_PROCS-${NUM_PROCS}_$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

# CPU 环境下使用 gloo 后端进行多进程通信模拟
torchrun $DISTRIBUTED_ARGS code/train.py \
    --distributed-backend gloo \
    # 2>&1 | tee logs/${logfile}.log
