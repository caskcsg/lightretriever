#!/bin/bash
BASE_DIR=$(dirname "$PWD")

TRAIL_NAME=$1
CKPT_NAME=$2
BENCHMARK_NAME=${BENCHMARK_NAME:-""}
TASK_NAME=${TASK_NAME:-""}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
CORPUS_CHUNK_SIZE=${CORPUS_CHUNK_SIZE:-100000}
TOPK=${TOPK:-1000}
INFERENCE_ARCH=${INFERENCE_ARCH:-"PytorchRPCExactSearchModel"}

EXTRA_ARGS=${EXTRA_ARGS:-""}  # Support override task-specific args

if [ "$CKPT_NAME" == "" ]; then
  MODEL_PATH=$BASE_DIR/results/${TRAIL_NAME}
  OUTPUT_PATH=$BASE_DIR/outputs/${TRAIL_NAME}/mteb
  LOG_DIR=$BASE_DIR/logs/${TRAIL_NAME}/mteb
else
  MODEL_PATH=$BASE_DIR/results/${TRAIL_NAME}/${CKPT_NAME}
  OUTPUT_PATH=$BASE_DIR/outputs/${TRAIL_NAME}-${CKPT_NAME}/mteb
  LOG_DIR=$BASE_DIR/logs/${TRAIL_NAME}-${CKPT_NAME}/mteb
fi

mkdir -p $OUTPUT_PATH
mkdir -p $LOG_DIR

# Distributed Command
CMD="torchrun "
CMD+=" --nnodes ${NNODES} "
CMD+=" --nproc_per_node ${NPROC_PER_NODE} "
CMD+=" --node_rank ${NODE_RANK} "
CMD+=" --master_addr ${MASTER_ADDR} "
CMD+=" --master_port ${MASTER_PORT} "

ARGS=""
ARGS+=" --model_name_or_path $MODEL_PATH "
if [ ! "$BENCHMARK_NAME" == "" ]; then
  ARGS+=" --benchmark_name ${BENCHMARK_NAME} "
fi
if [ ! "$TASK_NAME" == "" ]; then
  ARGS+=" --task_name ${TASK_NAME} "
fi
ARGS+=" --top_k $TOPK "
ARGS+=" --inference_arch $INFERENCE_ARCH "
ARGS+=" --output_dir $OUTPUT_PATH "
ARGS+=" --batch_size $EVAL_BATCH_SIZE "
ARGS+=" --corpus_chunk_size $CORPUS_CHUNK_SIZE "

set -x

# Test
$CMD evaluate_mteb.py \
  $ARGS \
  $MODEL_KWARGS \
  $EXTRA_ARGS \
  |& tee -a $LOG_DIR/eval-${BENCHMARK_NAME}-${TASK_NAME}-rank${NODE_RANK}-${NNODES}.log
