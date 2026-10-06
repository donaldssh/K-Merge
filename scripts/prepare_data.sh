#!/bin/bash

# run this script using "nohup bash scripts/prepare_data.sh > logs/prepare_data.txt 2>&1 &"

conda activate kmerge

mkdir -p logs

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NUM_GPUS=${#GPUS[@]}

run() {
  # $1: nameref to an array of "name|args" entries
  local -n entries=$1
  declare -A allocations
  local i=0
  for entry in "${entries[@]}"; do
    local gpu=${GPUS[$((i % NUM_GPUS))]}
    allocations[${gpu}]="${allocations[${gpu}]}"$'\n'"${entry}"
    i=$((i + 1))
  done

  for gpu in "${GPUS[@]}"; do
    (
    while IFS= read -r entry; do
      [ -z "$entry" ] && continue
      name="${entry%%|*}"
      args="${entry#*|}"
      (
      dt=$(date '+%d/%m/%Y %H:%M:%S')
      echo "Start time: $dt"
      CUDA_VISIBLE_DEVICES=$gpu python process_data.py ${args}
      dt=$(date '+%d/%m/%Y %H:%M:%S')
      echo "End time: $dt"
      ) > logs/${name}.txt 2>&1
    done <<< "${allocations[${gpu}]}"
    ) &
  done
  wait
}

# 1) English source data

phase1=(
  "content_rephrasing_tone_professional_prep_v1|--dataset content_rephrasing --tone professional"
  "content_rephrasing_tone_casual_prep_v1|--dataset content_rephrasing --tone casual"
  "content_rephrasing_tone_witty_prep_v1|--dataset content_rephrasing --tone witty"
  "content_rephrasing_tone_paraphrase_prep_v1|--dataset content_rephrasing --tone paraphrase"
  "text_correction_prep_v1|--dataset text-correction"
)
run phase1

# 2) machine translation into the other 7 languages 

TRANSLATE_LANGS=(de es fr it ja ko zh)
TRANSLATE_DATASETS=(persona-chat-synthetic squad samsum)
TONES=(professional casual witty paraphrase)

phase2=()
for dataset in "${TRANSLATE_DATASETS[@]}"; do
  for lang in "${TRANSLATE_LANGS[@]}"; do
    phase2+=("${dataset}_m100_${lang}_v1|--dataset ${dataset} --target_language ${lang}")
  done
done
for tone in "${TONES[@]}"; do
  for lang in "${TRANSLATE_LANGS[@]}"; do
    phase2+=("content_rephrasing_${tone}_m100_${lang}_v1|--dataset content_rephrasing --tone ${tone} --target_language ${lang}")
  done
done
run phase2
