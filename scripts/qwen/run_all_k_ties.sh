#!/bin/bash

# nohup bash continual_merging_scripts/qwen/run_all_k_ties.sh > logs/all_k_ties.txt 2>&1 &

# Runs every (k_lora_budget, seed) combination for "ties" merging method

MODEL_NAME_OR_PATH=Qwen/Qwen2.5-1.5B-Instruct

STORED_MODEL_NAME=qwen2515b_loras/correction_attn_lora_qwen2515b_ob_de_v0/

LORA_MERGE_MODULES=(
  "qwen2515b_loras/correction_attn_lora_qwen2515b_ob_de_v0/"
  "qwen2515b_loras/correction_attn_lora_qwen2515b_ob_en_v0/"
  "qwen2515b_loras/correction_attn_lora_qwen2515b_ob_es_v0/"
  "qwen2515b_loras/correction_attn_lora_qwen2515b_ob_fr_v0/"
  "qwen2515b_loras/correction_attn_lora_qwen2515b_ob_it_v0/"
  "qwen2515b_loras/correction_attn_lora_qwen2515b_ob_ja_v0/"
  "qwen2515b_loras/correction_attn_lora_qwen2515b_ob_ko_v0/"
  "qwen2515b_loras/correction_attn_lora_qwen2515b_ob_zh_v0/"
  "qwen2515b_loras/cr_attn_lora_qwen2515b_ob_generic_de_v0/"
  "qwen2515b_loras/cr_attn_lora_qwen2515b_ob_generic_en_v0/"
  "qwen2515b_loras/cr_attn_lora_qwen2515b_ob_generic_es_v0/"
  "qwen2515b_loras/cr_attn_lora_qwen2515b_ob_generic_fr_v0/"
  "qwen2515b_loras/cr_attn_lora_qwen2515b_ob_generic_it_v0/"
  "qwen2515b_loras/cr_attn_lora_qwen2515b_ob_generic_ja_v0/"
  "qwen2515b_loras/cr_attn_lora_qwen2515b_ob_generic_ko_v0/"
  "qwen2515b_loras/cr_attn_lora_qwen2515b_ob_generic_zh_v0/"
  "qwen2515b_loras/qa_attn_lora_qwen2515b_ob_de_v0/"
  "qwen2515b_loras/qa_attn_lora_qwen2515b_ob_en_v0/"
  "qwen2515b_loras/qa_attn_lora_qwen2515b_ob_es_v0/"
  "qwen2515b_loras/qa_attn_lora_qwen2515b_ob_fr_v0/"
  "qwen2515b_loras/qa_attn_lora_qwen2515b_ob_it_v0/"
  "qwen2515b_loras/qa_attn_lora_qwen2515b_ob_ja_v0/"
  "qwen2515b_loras/qa_attn_lora_qwen2515b_ob_ko_v0/"
  "qwen2515b_loras/qa_attn_lora_qwen2515b_ob_zh_v0/"
  "qwen2515b_loras/rp_attn_lora_qwen2515b_ob_de_v0/"
  "qwen2515b_loras/rp_attn_lora_qwen2515b_ob_en_v0/"
  "qwen2515b_loras/rp_attn_lora_qwen2515b_ob_es_v0/"
  "qwen2515b_loras/rp_attn_lora_qwen2515b_ob_fr_v0/"
  "qwen2515b_loras/rp_attn_lora_qwen2515b_ob_it_v0/"
  "qwen2515b_loras/rp_attn_lora_qwen2515b_ob_ja_v0/"
  "qwen2515b_loras/rp_attn_lora_qwen2515b_ob_ko_v0/"
  "qwen2515b_loras/rp_attn_lora_qwen2515b_ob_zh_v0/"
  "qwen2515b_loras/sum_attn_lora_qwen2515b_ob_de_v0/"
  "qwen2515b_loras/sum_attn_lora_qwen2515b_ob_en_v0/"
  "qwen2515b_loras/sum_attn_lora_qwen2515b_ob_es_v0/"
  "qwen2515b_loras/sum_attn_lora_qwen2515b_ob_fr_v0/"
  "qwen2515b_loras/sum_attn_lora_qwen2515b_ob_it_v0/"
  "qwen2515b_loras/sum_attn_lora_qwen2515b_ob_ja_v0/"
  "qwen2515b_loras/sum_attn_lora_qwen2515b_ob_ko_v0/"
  "qwen2515b_loras/sum_attn_lora_qwen2515b_ob_zh_v0/"
)

COMMON_ARGS="--model_name_or_path=${MODEL_NAME_OR_PATH} --dataset=content_rephrasing --shuffle_loras --stored_model_name=${STORED_MODEL_NAME} --lora_merge_strategy=ties --lora_merge_modules ${LORA_MERGE_MODULES[*]} --tone=generic --do_predict"

K_VALUES=(1 2 3 4 5 6 7 8)
SEEDS=(0 22 44)

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NUM_GPUS=${#GPUS[@]}

mkdir -p logs

declare -A scenarios
declare -A allocations

i=0
for k in "${K_VALUES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    name="k_${k}_seed_${seed}_ties"
    scenarios[${name}]="--k_lora_budget=${k} --experiment_name=${name} --seed=${seed}"

    gpu=${GPUS[$((i % NUM_GPUS))]}
    allocations[${gpu}]="${allocations[${gpu}]} ${name}"
    i=$((i + 1))
  done
done

for gpu in "${GPUS[@]}"; do
  (
  for name in ${allocations[${gpu}]}; do
    (
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "Start time: $dt"
    CUDA_VISIBLE_DEVICES=$gpu python run_experiments.py ${COMMON_ARGS} ${scenarios[${name}]}
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "End time: $dt"
    ) > logs/${name}.txt 2>&1
  done
  ) &
done

wait
