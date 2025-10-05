#!/bin/bash
# @Author  :   Ma (Ma787639046@outlook.com)
BASE_DIR=$(dirname "$PWD")  # Working dictionary

DOMAIN_CONFIG_PATH=$BASE_DIR/config/data/exp-m.json   # Data sampling configs. Defines train sets and sampling weights.
TRAIL_NAME=lightretriever-llama3.1-8b                 # Define a trail name. Fine-tuned retriever will be output in $OUTPUT_DIR
SEED=42
TRAIN_N_PASSAGES=8          # TRAIN_N_PASSAGES is composed of `1 Positive Passage + (n-1) Negative Passages`

# Batch Size & Steps
TOTAL_BATCH_SIZE=$((128))   # Total train batch size
REAL_BATCH_SIZE_PER_GPU=$(($TOTAL_BATCH_SIZE/$NPROC_PER_NODE/$NNODES))
MAX_STEPS=12000             # Max steps for fine-tune
SAVE_STEPS=2000             # Interval of saving steps. Works with SAVE_STRATEGY="steps"
SAVE_TOTAL_LIMIT=6          # Limits of saved checkpoints
SAVE_STRATEGY="no"          # Huggingface Trainer save strategy. Choose among 'steps', 'epoch', 'no'

MODEL_PATH=meta-llama/Llama-3.1-8B        # Base Model to initialize from
OUTPUT_DIR=$BASE_DIR/results/$TRAIL_NAME  # Path to save fine-tuned retrievers
LOG_DIR=$BASE_DIR/logs/$TRAIL_NAME/dpr    # Path to log folder
mkdir -p $LOG_DIR

# Global Model Arguments
MODEL_KWARGS=""
MODEL_KWARGS+=" --model_type HybridModel "    # HybridModel: supports multi-embedding type fine-tuning; EncoderModel: only symmetric dense fine-tuning
# MODEL_KWARGS+=" --untie_encoder "           # Whether to seperate the weight sharing of query-document encoders.
MODEL_KWARGS+=" --score_function cos_sim "    # Dense embedding similarity function. Choose among cos_sim / dot.
MODEL_KWARGS+=" --q_max_len 512 "             # Query max sequence length
MODEL_KWARGS+=" --p_max_len 512 "             # Document max sequence length
MODEL_KWARGS+=" --bf16 "
MODEL_KWARGS+=" --add_prompt "                # Whether to add query prompt. Training prompt defined in `src/lightretriever/utils/prompts.py`
MODEL_KWARGS+=" --prompt_type e5 "            # Prompt types. Choosing among 'e5', 'instructor', 'bge'
# MODEL_KWARGS+=" --attn_implementation sdpa "  # Attention Impl: Default to flash_attention_2
MODEL_KWARGS+=" --cumulative_seq "            # Cumulative sequence packing. Speedup training/inferencing by eliminating padding area when forwarding the LLM, and conveniently **nonperceptible** to pooling logics.
MODEL_KWARGS+=" --liger_kernel "              # Fused Triton-based Liger Kernel to speedup training/inferencing.
# MODEL_KWARGS+=" --enable_bidirectional_attention "    # Hacking bi-directional attention for causal LLM.

# Special Tokens
MODEL_KWARGS+=" --edit_tokenizer_normalizers "      # Modify Huggingface tokenizer' normalizer, to support lower case.
MODEL_KWARGS+=" --lowercase "                       # Lower case all input texts to promote lexical overlap for sparse retrieval.
MODEL_KWARGS+=" --edit_tokenizer_post_processor "   # Modify Huggingface tokenizer' postprocessor, to support add / format with special tokens like EOS
MODEL_KWARGS+=" --add_bos_num 1 "                   # Ensure one <bos> in front of tokenized inputs
MODEL_KWARGS+=" --add_eos_num 1 "                   # Ensure one <eos> at the end of tokenized inputs
MODEL_KWARGS+=" --add_sep_token "

# ** Hybrid Model Arguments **
MODEL_KWARGS+=" --hybrid_use_dense_vector "         # Symmetric dense vector
# MODEL_KWARGS+=" --hybrid_use_sparse_vector "      # Symmetric sparse vector
MODEL_KWARGS+=" --hybrid_use_emb_vector "           # Asymmetric dense vector
MODEL_KWARGS+=" --hybrid_use_token_id_vector "      # Asymmetric sparse vector

# >> Non-contextual query embedding as emb_vector
MODEL_KWARGS+=" --noncontextual_query_embedding "   # Enable: Use LightRetriever's asymmetric dense vector; Disblae: Use LLM's Embedding Layer.

# >> Aggregation
MODEL_KWARGS+=" --sparse_use_max_aggregation "      # Sparse vector: Max Aggregation

# >> ReLU, Log Saturation
MODEL_KWARGS+=" --sparse_use_relu "                 # Sparse vector: ReLU
MODEL_KWARGS+=" --sparse_use_log_saturation "       # Sparse vector: log(x + 1)

# >> Pooling
MODEL_KWARGS+=" --pooling_strategy lasttoken "      # Dense vector: EOS Pooling

# Set General Model Arguments
export MODEL_KWARGS=$MODEL_KWARGS

##########################
# Common Fine-tuning Args
##########################
# Distributed Command
CMD="accelerate launch "
CMD+=" --num_machines ${NNODES} "
CMD+=" --machine_rank ${NODE_RANK} "
CMD+=" --num_processes $((NNODES*NPROC_PER_NODE)) "
CMD+=" --main_process_ip ${MASTER_ADDR} "
CMD+=" --main_process_port ${MASTER_PORT} "

# ** Deepspeed / FSDP **
# CMD+=" --config_file $BASE_DIR/config/ddp.yaml "
# CMD+=" --config_file $BASE_DIR/config/fsdp_shard_grad_op.yaml "
# CMD+=" --config_file $BASE_DIR/config/fsdp_hybrid_shard.yaml "
CMD+=" --config_file $BASE_DIR/config/fsdp_full_shard.yaml "
# CMD+=" --config_file $BASE_DIR/config/fsdp_v2_shard_grad_op.yaml "
# CMD+=" --config_file $BASE_DIR/config/fsdp_v2_full_shard.yaml "
# CMD+=" --config_file $BASE_DIR/config/ds_stage0.yaml "
# CMD+=" --config_file $BASE_DIR/config/ds_stage1.yaml "
# CMD+=" --config_file $BASE_DIR/config/ds_stage2.yaml "
# CMD+=" --config_file $BASE_DIR/config/ds_stage3.yaml "
CMD+=" -m lightretriever.finetune.fit "

# Data Arguments
DATA_ARGS=""
DATA_ARGS+=" --domain_config_path $DOMAIN_CONFIG_PATH "   # Data sampling configs. Defines train sets and sampling weights.
DATA_ARGS+=" --preprocessed_dir $BASE_DIR/data/train "    # Path to train set root folder
DATA_ARGS+=" --homogenous_batch "         # Homogenous batching samples a batch with only **one dataset**. Important to ensure negative quality.
DATA_ARGS+=" --pad_to_multiple_of 8 "
# DATA_ARGS+=" --pad_to_max_length "

# Training Arguments
TRAIN_ARGS=""
TRAIN_ARGS+=" --do_train "
TRAIN_ARGS+=" --lora "              # Enable LoRA
TRAIN_ARGS+=" --lora_r 16 "         # LoRA r
TRAIN_ARGS+=" --lora_alpha 32 "     # LoRA alpha
TRAIN_ARGS+=" --lora_dropout 0.1 "  # LoRA dropout
TRAIN_ARGS+=" --save_strategy $SAVE_STRATEGY "
TRAIN_ARGS+=" --save_steps $SAVE_STEPS "
TRAIN_ARGS+=" --save_total_limit $SAVE_TOTAL_LIMIT "
TRAIN_ARGS+=" --save_only_model "         # Only save model, not optimizer, scheduler, etc.
TRAIN_ARGS+=" --logging_steps 2 "
TRAIN_ARGS+=" --warmup_steps 100 "
TRAIN_ARGS+=" --per_device_train_batch_size $REAL_BATCH_SIZE_PER_GPU "
TRAIN_ARGS+=" --learning_rate 2e-5 "
TRAIN_ARGS+=" --min_lr_ratio 0.1 "
TRAIN_ARGS+=" --lr_scheduler_type cosine "
TRAIN_ARGS+=" --max_steps $MAX_STEPS "
TRAIN_ARGS+=" --temperature 0.02 "        # Dense temperature
TRAIN_ARGS+=" --sparse_temperature 1.0 "  # Sparse temperature
TRAIN_ARGS+=" --train_n_passages $TRAIN_N_PASSAGES "
TRAIN_ARGS+=" --negatives_x_device "      # Gather cross-batch negatives from other ranks
TRAIN_ARGS+=" --seed $SEED "
TRAIN_ARGS+=" --dataloader_num_workers 4 "
TRAIN_ARGS+=" --optim adamw_torch_fused "
TRAIN_ARGS+=" --weight_decay 0.01 "
TRAIN_ARGS+=" --gradient_checkpointing "
# TRAIN_ARGS+=" --grad_cache "            # Enable GradCache when GPU OOM
# TRAIN_ARGS+=" --no_sync_except_last "
# TRAIN_ARGS+=" --gc_q_chunk_size 32 "
# TRAIN_ARGS+=" --gc_p_chunk_size 32 "

# >> Distillation
# TRAIN_ARGS+=" --distillation "              # Distill pre-computed `ce_scores` (Cross Encoder Scores) from training data.
# TRAIN_ARGS+=" --distill_temperature 1.0 "   # Distillation temperature
# TRAIN_ARGS+=" --distill_coef 1.0 "          # Distillation coef

# >> MRL
# TRAIN_ARGS+=" --matryoshka_dims 128 256 512 1024 2048 4096 "

## Distillation: Imbalanced Dense Emb Vec Scores -> Full Dense Scores
# TRAIN_ARGS+=" --emb_den_scores_distillation "
# TRAIN_ARGS+=" --emb_reps_distill_coef 1.0 "

## Distillation: Imbalanced Sparse Token id Vec Scores -> Full Dense Scores
# TRAIN_ARGS+=" --tok_den_scores_distillation "
# TRAIN_ARGS+=" --tok_reps_distill_coef 1.0 "

# ** Hybrid Train Arguments **
# >> Regulator
TRAIN_ARGS+=" --add_flops "
TRAIN_ARGS+=" --q_norm_loss_factor 1e-3 "
TRAIN_ARGS+=" --p_norm_loss_factor 1e-3 "

# >> Regulation Factor Scheduler
TRAIN_ARGS+=" --use_reg_scheduler "
TRAIN_ARGS+=" --reg_t 4000 "
TRAIN_ARGS+=" --reg_type quadratic "

set -ex
##########################
# Launch Fine-tuning 
##########################
$CMD \
  --model_name_or_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --report_to tensorboard \
  $DATA_ARGS \
  $TRAIN_ARGS \
  $MODEL_KWARGS \
  |& tee $LOG_DIR/finetune_rank${NODE_RANK}-${NNODES}.log

