#!/bin/bash
iter_num=3
for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        MODEL="meta-llama/Llama-3.2-3B-Instruct"
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/Llama-3.2-3B-Instruct-SPPO-Iter${i}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"
    OUT="data-llama-3.2-3b-instruct-sppo-iter${i}"

    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline.sh --model $MODEL --iter $i --dataset "synthetic_data_llama-3.2-3b-instruct-sppo-iter${i}_score" --output_dir $OUTPUT_DIR --num 1
done
