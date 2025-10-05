# Evaluation
This folder holds evaluation guidelines for BeIR & CMTEB-Retrieval. Our evaluation pipeline is developed based on [Distributed RPC Framework](https://pytorch.org/docs/stable/rpc.html), which naturally supports multi-node, multi-GPU encoding.
<!-- akins the pattern of [one producer - multiple consumers (workers)](https://en.wikipedia.org/wiki/Producer–consumer_problem) and  -->

## Evaluation Scripts
First, please set corresponding model arguments:

```bash
# Global Model Arguments
MODEL_KWARGS=""

# >> Model Settings
MODEL_KWARGS+=" --model_type HybridModel "      # HybridModel: supports multi-embedding types


# >> Embedding Type Selection
# Note: MTEB only support presenting evaluation metrics of one embedding type at a time, thus you can choose combinations below: 
#       1. Only Symmetric dense vector
#       2. Only Asymmetric dense vector
#       3. Only Asymmetric sparse vector
#       4. Enable Both Asymmetric dense vector and Asymmetric sparse vector for Hybrid Retrieval

# Symmetric dense vector
MODEL_KWARGS+=" --hybrid_use_dense_vector "

# Asymmetric dense vector
# MODEL_KWARGS+=" --hybrid_use_emb_vector "
# MODEL_KWARGS+=" --noncontextual_query_embedding "   # Enable: Use LightRetriever's asymmetric dense vector; Disblae: Use LLM's Embedding Layer.

# Asymmetric sparse vector
# MODEL_KWARGS+=" --hybrid_use_token_id_vector "


# >> Data Args
MODEL_KWARGS+=" --bf16 "            # Bfloat16 training / inferencing (Mix-precision w/ auto-cast)
MODEL_KWARGS+=" --q_max_len 512 "   # Query max length
MODEL_KWARGS+=" --p_max_len 512 "   # Passage max length

# >> Prompts (Eval prompt defined in `eval/prompts.py`)
MODEL_KWARGS+=" --add_prompt "      # Whether to add prompt in front of the queries. 
MODEL_KWARGS+=" --prompt_type e5 "  # Here we follow the prompt settings of Mistral-E5

# >> Pooling Methods
MODEL_KWARGS+=" --score_function cos_sim "      # Dense embedding cosine similarity
MODEL_KWARGS+=" --pooling_strategy lasttoken "  # Dense embedding last token (</eos>) pooling
MODEL_KWARGS+=" --sparse_use_max_aggregation "      # Sparse vector: Max Aggregation
MODEL_KWARGS+=" --sparse_use_relu "                 # Sparse vector: ReLU
MODEL_KWARGS+=" --sparse_use_log_saturation "       # Sparse vector: log(x + 1)

# >> (Optional) Speed up
MODEL_KWARGS+=" --cumulative_seq "  # Cumulative sequence packing. Speedup training/inferencing by eliminating padding area when forwarding the LLM, and conveniently **nonperceptible** to pooling logics.
MODEL_KWARGS+=" --liger_kernel "    # Fused Triton-based Liger Kernel to speedup training/inferencing.

# Export Above Model Arguments to Env
export MODEL_KWARGS=$MODEL_KWARGS
```

Assume the retriever (folder name `TRAIL_NAME`) is located in `lightretriever/results/$TRAIL_NAME`. Please pick the following reference commands as you wish to evaluate:

```bash
TASK_NAME=ArguAna bash call_evaluate_mteb.sh $TRAIL_NAME
BENCHMARK_NAME=BEIR bash call_evaluate_mteb.sh $TRAIL_NAME
BENCHMARK_NAME=CMTEB-R bash call_evaluate_mteb.sh $TRAIL_NAME
```

## Acknowledgement
Our evaluation pipeline is adpoted from [tDRO](https://github.com/ma787639046/tdro), which is developed based on [MTEB](https://github.com/embeddings-benchmark/mteb).
