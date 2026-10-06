#!/bin/bash

# run this script using nohup bash scripts/loras_qwen.sh > logs/experiments.txt 2>&1 &

conda activate kmerge

declare -A arguments

model_name_or_path_qwen="Qwen/Qwen2.5-1.5B-Instruct"

# Qwen - en
arguments["rp_zero_shot_qwen2515b_ob_en_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["rp_attn_lora_qwen2515b_ob_en_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --stored_model_name rp_attn_lora_qwen2515b_ob_en_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_qwen2515b_ob_en_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["qa_attn_lora_qwen2515b_ob_en_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --model_name_or_path $model_name_or_path_qwen --stored_model_name qa_attn_lora_qwen2515b_ob_en_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_qwen2515b_ob_en_v0"]="--dataset samsum --zero_shot --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["sum_attn_lora_qwen2515b_ob_en_v0"]="--dataset samsum --attn_lora --model_name_or_path $model_name_or_path_qwen --stored_model_name sum_attn_lora_qwen2515b_ob_en_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_qwen2515b_ob_en_v0"]="--dataset content_rephrasing --tone generic --zero_shot --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["cr_attn_lora_qwen2515b_ob_generic_en_v0"]="--dataset content_rephrasing --tone generic --attn_lora --stored_model_name cr_attn_lora_qwen2515b_ob_generic_en_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_qwen2515b_ob_en_v0"]="--dataset text-correction --zero_shot --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["correction_attn_lora_qwen2515b_ob_en_v0"]="--dataset text-correction --attn_lora --stored_model_name correction_attn_lora_qwen2515b_ob_en_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"

# Qwen - de
arguments["rp_zero_shot_qwen2515b_ob_de_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["rp_attn_lora_qwen2515b_ob_de_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language de --stored_model_name rp_attn_lora_qwen2515b_ob_de_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_qwen2515b_ob_de_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["qa_attn_lora_qwen2515b_ob_de_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language de --model_name_or_path $model_name_or_path_qwen --stored_model_name qa_attn_lora_qwen2515b_ob_de_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_qwen2515b_ob_de_v0"]="--dataset samsum --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["sum_attn_lora_qwen2515b_ob_de_v0"]="--dataset samsum --attn_lora --dataset_language de --model_name_or_path $model_name_or_path_qwen --stored_model_name sum_attn_lora_qwen2515b_ob_de_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_qwen2515b_ob_de_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["cr_attn_lora_qwen2515b_ob_generic_de_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language de --stored_model_name cr_attn_lora_qwen2515b_ob_generic_de_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_qwen2515b_ob_de_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["correction_attn_lora_qwen2515b_ob_de_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language de --stored_model_name correction_attn_lora_qwen2515b_ob_de_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Qwen - es
arguments["rp_zero_shot_qwen2515b_ob_es_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["rp_attn_lora_qwen2515b_ob_es_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language es --stored_model_name rp_attn_lora_qwen2515b_ob_es_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_qwen2515b_ob_es_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["qa_attn_lora_qwen2515b_ob_es_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language es --model_name_or_path $model_name_or_path_qwen --stored_model_name qa_attn_lora_qwen2515b_ob_es_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_qwen2515b_ob_es_v0"]="--dataset samsum --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["sum_attn_lora_qwen2515b_ob_es_v0"]="--dataset samsum --attn_lora --dataset_language es --model_name_or_path $model_name_or_path_qwen --stored_model_name sum_attn_lora_qwen2515b_ob_es_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_qwen2515b_ob_es_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["cr_attn_lora_qwen2515b_ob_generic_es_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language es --stored_model_name cr_attn_lora_qwen2515b_ob_generic_es_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_qwen2515b_ob_es_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["correction_attn_lora_qwen2515b_ob_es_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language es --stored_model_name correction_attn_lora_qwen2515b_ob_es_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Qwen - fr
arguments["rp_zero_shot_qwen2515b_ob_fr_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["rp_attn_lora_qwen2515b_ob_fr_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language fr --stored_model_name rp_attn_lora_qwen2515b_ob_fr_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_qwen2515b_ob_fr_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["qa_attn_lora_qwen2515b_ob_fr_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language fr --model_name_or_path $model_name_or_path_qwen --stored_model_name qa_attn_lora_qwen2515b_ob_fr_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_qwen2515b_ob_fr_v0"]="--dataset samsum --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["sum_attn_lora_qwen2515b_ob_fr_v0"]="--dataset samsum --attn_lora --dataset_language fr --model_name_or_path $model_name_or_path_qwen --stored_model_name sum_attn_lora_qwen2515b_ob_fr_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_qwen2515b_ob_fr_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["cr_attn_lora_qwen2515b_ob_generic_fr_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language fr --stored_model_name cr_attn_lora_qwen2515b_ob_generic_fr_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_qwen2515b_ob_fr_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["correction_attn_lora_qwen2515b_ob_fr_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language fr --stored_model_name correction_attn_lora_qwen2515b_ob_fr_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Qwen - it
arguments["rp_zero_shot_qwen2515b_ob_it_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["rp_attn_lora_qwen2515b_ob_it_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language it --stored_model_name rp_attn_lora_qwen2515b_ob_it_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_qwen2515b_ob_it_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["qa_attn_lora_qwen2515b_ob_it_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language it --model_name_or_path $model_name_or_path_qwen --stored_model_name qa_attn_lora_qwen2515b_ob_it_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_qwen2515b_ob_it_v0"]="--dataset samsum --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["sum_attn_lora_qwen2515b_ob_it_v0"]="--dataset samsum --attn_lora --dataset_language it --model_name_or_path $model_name_or_path_qwen --stored_model_name sum_attn_lora_qwen2515b_ob_it_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_qwen2515b_ob_it_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["cr_attn_lora_qwen2515b_ob_generic_it_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language it --stored_model_name cr_attn_lora_qwen2515b_ob_generic_it_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_qwen2515b_ob_it_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["correction_attn_lora_qwen2515b_ob_it_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language it --stored_model_name correction_attn_lora_qwen2515b_ob_it_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Qwen - ja
arguments["rp_zero_shot_qwen2515b_ob_ja_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["rp_attn_lora_qwen2515b_ob_ja_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language ja --stored_model_name rp_attn_lora_qwen2515b_ob_ja_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_qwen2515b_ob_ja_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["qa_attn_lora_qwen2515b_ob_ja_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language ja --model_name_or_path $model_name_or_path_qwen --stored_model_name qa_attn_lora_qwen2515b_ob_ja_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_qwen2515b_ob_ja_v0"]="--dataset samsum --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["sum_attn_lora_qwen2515b_ob_ja_v0"]="--dataset samsum --attn_lora --dataset_language ja --model_name_or_path $model_name_or_path_qwen --stored_model_name sum_attn_lora_qwen2515b_ob_ja_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_qwen2515b_ob_ja_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["cr_attn_lora_qwen2515b_ob_generic_ja_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language ja --stored_model_name cr_attn_lora_qwen2515b_ob_generic_ja_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_qwen2515b_ob_ja_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["correction_attn_lora_qwen2515b_ob_ja_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language ja --stored_model_name correction_attn_lora_qwen2515b_ob_ja_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Qwen - ko
arguments["rp_zero_shot_qwen2515b_ob_ko_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["rp_attn_lora_qwen2515b_ob_ko_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language ko --stored_model_name rp_attn_lora_qwen2515b_ob_ko_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_qwen2515b_ob_ko_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["qa_attn_lora_qwen2515b_ob_ko_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language ko --model_name_or_path $model_name_or_path_qwen --stored_model_name qa_attn_lora_qwen2515b_ob_ko_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_qwen2515b_ob_ko_v0"]="--dataset samsum --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["sum_attn_lora_qwen2515b_ob_ko_v0"]="--dataset samsum --attn_lora --dataset_language ko --model_name_or_path $model_name_or_path_qwen --stored_model_name sum_attn_lora_qwen2515b_ob_ko_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_qwen2515b_ob_ko_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["cr_attn_lora_qwen2515b_ob_generic_ko_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language ko --stored_model_name cr_attn_lora_qwen2515b_ob_generic_ko_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_qwen2515b_ob_ko_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["correction_attn_lora_qwen2515b_ob_ko_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language ko --stored_model_name correction_attn_lora_qwen2515b_ob_ko_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Qwen - zh
arguments["rp_zero_shot_qwen2515b_ob_zh_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["rp_attn_lora_qwen2515b_ob_zh_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language zh --stored_model_name rp_attn_lora_qwen2515b_ob_zh_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_qwen2515b_ob_zh_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["qa_attn_lora_qwen2515b_ob_zh_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language zh --model_name_or_path $model_name_or_path_qwen --stored_model_name qa_attn_lora_qwen2515b_ob_zh_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_qwen2515b_ob_zh_v0"]="--dataset samsum --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["sum_attn_lora_qwen2515b_ob_zh_v0"]="--dataset samsum --attn_lora --dataset_language zh --model_name_or_path $model_name_or_path_qwen --stored_model_name sum_attn_lora_qwen2515b_ob_zh_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_qwen2515b_ob_zh_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["cr_attn_lora_qwen2515b_ob_generic_zh_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language zh --stored_model_name cr_attn_lora_qwen2515b_ob_generic_zh_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_qwen2515b_ob_zh_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_qwen --do_valid --do_predict"
arguments["correction_attn_lora_qwen2515b_ob_zh_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language zh --stored_model_name correction_attn_lora_qwen2515b_ob_zh_v0 --model_name_or_path $model_name_or_path_qwen --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

declare -A experiments_list

experiments_list[0]="rp_attn_lora_qwen2515b_ob_en_v0 cr_attn_lora_qwen2515b_ob_generic_de_v0 qa_attn_lora_qwen2515b_ob_fr_v0 correction_attn_lora_qwen2515b_ob_it_v0 sum_attn_lora_qwen2515b_ob_ko_v0"
experiments_list[1]="qa_attn_lora_qwen2515b_ob_en_v0 correction_attn_lora_qwen2515b_ob_de_v0 sum_attn_lora_qwen2515b_ob_fr_v0 rp_attn_lora_qwen2515b_ob_ja_v0 cr_attn_lora_qwen2515b_ob_generic_ko_v0"
experiments_list[2]="sum_attn_lora_qwen2515b_ob_en_v0 rp_attn_lora_qwen2515b_ob_es_v0 cr_attn_lora_qwen2515b_ob_generic_fr_v0 qa_attn_lora_qwen2515b_ob_ja_v0 correction_attn_lora_qwen2515b_ob_ko_v0"
experiments_list[3]="cr_attn_lora_qwen2515b_ob_generic_en_v0 qa_attn_lora_qwen2515b_ob_es_v0 correction_attn_lora_qwen2515b_ob_fr_v0 sum_attn_lora_qwen2515b_ob_ja_v0 rp_attn_lora_qwen2515b_ob_zh_v0"
experiments_list[4]="correction_attn_lora_qwen2515b_ob_en_v0 sum_attn_lora_qwen2515b_ob_es_v0 rp_attn_lora_qwen2515b_ob_it_v0 cr_attn_lora_qwen2515b_ob_generic_ja_v0 qa_attn_lora_qwen2515b_ob_zh_v0"
experiments_list[5]="rp_attn_lora_qwen2515b_ob_de_v0 cr_attn_lora_qwen2515b_ob_generic_es_v0 qa_attn_lora_qwen2515b_ob_it_v0 correction_attn_lora_qwen2515b_ob_ja_v0 sum_attn_lora_qwen2515b_ob_zh_v0"
experiments_list[6]="qa_attn_lora_qwen2515b_ob_de_v0 correction_attn_lora_qwen2515b_ob_es_v0 sum_attn_lora_qwen2515b_ob_it_v0 rp_attn_lora_qwen2515b_ob_ko_v0 cr_attn_lora_qwen2515b_ob_generic_zh_v0"
experiments_list[7]="sum_attn_lora_qwen2515b_ob_de_v0 rp_attn_lora_qwen2515b_ob_fr_v0 cr_attn_lora_qwen2515b_ob_generic_it_v0 qa_attn_lora_qwen2515b_ob_ko_v0 correction_attn_lora_qwen2515b_ob_zh_v0"

for gpu in 0 1 2 3 4 5 6 7; do
  (
  for name in ${experiments_list[${gpu}]}; do
    (
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "Start time: $dt"
    CUDA_VISIBLE_DEVICES=$gpu python run_experiments.py --experiment_name ${name} ${arguments[${name}]}
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "End time: $dt"
    ) > logs/${name}.txt 2>&1
  done
  ) &
done