#!/bin/bash

# run this script using nohup bash scripts/loras_llama.sh > logs/experiments.txt 2>&1 &


conda activate kmerge

declare -A arguments

model_name_or_path_llama="meta-llama/Llama-3.2-1B-Instruct"

# Llama - en
arguments["rp_zero_shot_l32_1b_tc_en_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["rp_attn_lora_l32_1b_tc_en_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --stored_model_name rp_attn_lora_l32_1b_tc_en_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_l32_1b_tc_en_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["qa_attn_lora_l32_1b_tc_en_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --model_name_or_path $model_name_or_path_llama --stored_model_name qa_attn_lora_l32_1b_tc_en_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_l32_1b_tc_en_v0"]="--dataset samsum --zero_shot --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["sum_attn_lora_l32_1b_tc_en_v0"]="--dataset samsum --attn_lora --model_name_or_path $model_name_or_path_llama --stored_model_name sum_attn_lora_l32_1b_tc_en_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_l32_1b_tc_en_v0"]="--dataset content_rephrasing --tone generic --zero_shot --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["cr_attn_lora_l32_1b_tc_generic_en_v0"]="--dataset content_rephrasing --tone generic --attn_lora --stored_model_name cr_attn_lora_l32_1b_tc_generic_en_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_l32_1b_tc_en_v0"]="--dataset text-correction --zero_shot --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["correction_attn_lora_l32_1b_tc_en_v0"]="--dataset text-correction --attn_lora --stored_model_name correction_attn_lora_l32_1b_tc_en_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"

# Llama - de
arguments["rp_zero_shot_l32_1b_tc_de_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["rp_attn_lora_l32_1b_tc_de_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language de --stored_model_name rp_attn_lora_l32_1b_tc_de_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_l32_1b_tc_de_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["qa_attn_lora_l32_1b_tc_de_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language de --model_name_or_path $model_name_or_path_llama --stored_model_name qa_attn_lora_l32_1b_tc_de_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_l32_1b_tc_de_v0"]="--dataset samsum --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["sum_attn_lora_l32_1b_tc_de_v0"]="--dataset samsum --attn_lora --dataset_language de --model_name_or_path $model_name_or_path_llama --stored_model_name sum_attn_lora_l32_1b_tc_de_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_l32_1b_tc_de_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["cr_attn_lora_l32_1b_tc_generic_de_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language de --stored_model_name cr_attn_lora_l32_1b_tc_generic_de_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_l32_1b_tc_de_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language de --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["correction_attn_lora_l32_1b_tc_de_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language de --stored_model_name correction_attn_lora_l32_1b_tc_de_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Llama - es
arguments["rp_zero_shot_l32_1b_tc_es_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["rp_attn_lora_l32_1b_tc_es_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language es --stored_model_name rp_attn_lora_l32_1b_tc_es_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_l32_1b_tc_es_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["qa_attn_lora_l32_1b_tc_es_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language es --model_name_or_path $model_name_or_path_llama --stored_model_name qa_attn_lora_l32_1b_tc_es_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_l32_1b_tc_es_v0"]="--dataset samsum --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["sum_attn_lora_l32_1b_tc_es_v0"]="--dataset samsum --attn_lora --dataset_language es --model_name_or_path $model_name_or_path_llama --stored_model_name sum_attn_lora_l32_1b_tc_es_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_l32_1b_tc_es_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["cr_attn_lora_l32_1b_tc_generic_es_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language es --stored_model_name cr_attn_lora_l32_1b_tc_generic_es_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_l32_1b_tc_es_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language es --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["correction_attn_lora_l32_1b_tc_es_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language es --stored_model_name correction_attn_lora_l32_1b_tc_es_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Llama - fr
arguments["rp_zero_shot_l32_1b_tc_fr_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["rp_attn_lora_l32_1b_tc_fr_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language fr --stored_model_name rp_attn_lora_l32_1b_tc_fr_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_l32_1b_tc_fr_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["qa_attn_lora_l32_1b_tc_fr_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language fr --model_name_or_path $model_name_or_path_llama --stored_model_name qa_attn_lora_l32_1b_tc_fr_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_l32_1b_tc_fr_v0"]="--dataset samsum --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["sum_attn_lora_l32_1b_tc_fr_v0"]="--dataset samsum --attn_lora --dataset_language fr --model_name_or_path $model_name_or_path_llama --stored_model_name sum_attn_lora_l32_1b_tc_fr_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_l32_1b_tc_fr_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["cr_attn_lora_l32_1b_tc_generic_fr_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language fr --stored_model_name cr_attn_lora_l32_1b_tc_generic_fr_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_l32_1b_tc_fr_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language fr --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["correction_attn_lora_l32_1b_tc_fr_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language fr --stored_model_name correction_attn_lora_l32_1b_tc_fr_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Llama - it
arguments["rp_zero_shot_l32_1b_tc_it_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["rp_attn_lora_l32_1b_tc_it_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language it --stored_model_name rp_attn_lora_l32_1b_tc_it_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_l32_1b_tc_it_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["qa_attn_lora_l32_1b_tc_it_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language it --model_name_or_path $model_name_or_path_llama --stored_model_name qa_attn_lora_l32_1b_tc_it_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_l32_1b_tc_it_v0"]="--dataset samsum --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["sum_attn_lora_l32_1b_tc_it_v0"]="--dataset samsum --attn_lora --dataset_language it --model_name_or_path $model_name_or_path_llama --stored_model_name sum_attn_lora_l32_1b_tc_it_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_l32_1b_tc_it_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["cr_attn_lora_l32_1b_tc_generic_it_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language it --stored_model_name cr_attn_lora_l32_1b_tc_generic_it_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_l32_1b_tc_it_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language it --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["correction_attn_lora_l32_1b_tc_it_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language it --stored_model_name correction_attn_lora_l32_1b_tc_it_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Llama - ja
arguments["rp_zero_shot_l32_1b_tc_ja_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["rp_attn_lora_l32_1b_tc_ja_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language ja --stored_model_name rp_attn_lora_l32_1b_tc_ja_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_l32_1b_tc_ja_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["qa_attn_lora_l32_1b_tc_ja_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language ja --model_name_or_path $model_name_or_path_llama --stored_model_name qa_attn_lora_l32_1b_tc_ja_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_l32_1b_tc_ja_v0"]="--dataset samsum --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["sum_attn_lora_l32_1b_tc_ja_v0"]="--dataset samsum --attn_lora --dataset_language ja --model_name_or_path $model_name_or_path_llama --stored_model_name sum_attn_lora_l32_1b_tc_ja_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_l32_1b_tc_ja_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["cr_attn_lora_l32_1b_tc_generic_ja_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language ja --stored_model_name cr_attn_lora_l32_1b_tc_generic_ja_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_l32_1b_tc_ja_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language ja --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["correction_attn_lora_l32_1b_tc_ja_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language ja --stored_model_name correction_attn_lora_l32_1b_tc_ja_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Llama - ko
arguments["rp_zero_shot_l32_1b_tc_ko_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["rp_attn_lora_l32_1b_tc_ko_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language ko --stored_model_name rp_attn_lora_l32_1b_tc_ko_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_l32_1b_tc_ko_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["qa_attn_lora_l32_1b_tc_ko_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language ko --model_name_or_path $model_name_or_path_llama --stored_model_name qa_attn_lora_l32_1b_tc_ko_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_l32_1b_tc_ko_v0"]="--dataset samsum --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["sum_attn_lora_l32_1b_tc_ko_v0"]="--dataset samsum --attn_lora --dataset_language ko --model_name_or_path $model_name_or_path_llama --stored_model_name sum_attn_lora_l32_1b_tc_ko_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_l32_1b_tc_ko_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["cr_attn_lora_l32_1b_tc_generic_ko_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language ko --stored_model_name cr_attn_lora_l32_1b_tc_generic_ko_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_l32_1b_tc_ko_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language ko --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["correction_attn_lora_l32_1b_tc_ko_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language ko --stored_model_name correction_attn_lora_l32_1b_tc_ko_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

# Llama - zh
arguments["rp_zero_shot_l32_1b_tc_zh_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["rp_attn_lora_l32_1b_tc_zh_v0"]="--dataset persona-chat-synthetic --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language zh --stored_model_name rp_attn_lora_l32_1b_tc_zh_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["qa_zero_shot_l32_1b_tc_zh_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["qa_attn_lora_l32_1b_tc_zh_v0"]="--dataset squad --max_eval_samples 1000 --max_test_samples 1000 --attn_lora --dataset_language zh --model_name_or_path $model_name_or_path_llama --stored_model_name qa_attn_lora_l32_1b_tc_zh_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["sum_zero_shot_l32_1b_tc_zh_v0"]="--dataset samsum --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["sum_attn_lora_l32_1b_tc_zh_v0"]="--dataset samsum --attn_lora --dataset_language zh --model_name_or_path $model_name_or_path_llama --stored_model_name sum_attn_lora_l32_1b_tc_zh_v0 --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05"
arguments["cr_generic_zero_shot_l32_1b_tc_zh_v0"]="--dataset content_rephrasing --tone generic --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["cr_attn_lora_l32_1b_tc_generic_zh_v0"]="--dataset content_rephrasing --tone generic --attn_lora --dataset_language zh --stored_model_name cr_attn_lora_l32_1b_tc_generic_zh_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"
arguments["correction_zero_shot_l32_1b_tc_zh_v0"]="--dataset text-correction-org-lang --zero_shot --dataset_language zh --model_name_or_path $model_name_or_path_llama --do_valid --do_predict"
arguments["correction_attn_lora_l32_1b_tc_zh_v0"]="--dataset text-correction-org-lang --attn_lora --dataset_language zh --stored_model_name correction_attn_lora_l32_1b_tc_zh_v0 --model_name_or_path $model_name_or_path_llama --do_train --do_valid --do_predict --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --gradient_accumulation_steps 4"

declare -A experiments_list

experiments_list[0]="rp_attn_lora_l32_1b_tc_en_v0 cr_attn_lora_l32_1b_tc_generic_de_v0 qa_attn_lora_l32_1b_tc_fr_v0 correction_attn_lora_l32_1b_tc_it_v0 sum_attn_lora_l32_1b_tc_ko_v0"
experiments_list[1]="qa_attn_lora_l32_1b_tc_en_v0 correction_attn_lora_l32_1b_tc_de_v0 sum_attn_lora_l32_1b_tc_fr_v0 rp_attn_lora_l32_1b_tc_ja_v0 cr_attn_lora_l32_1b_tc_generic_ko_v0"
experiments_list[2]="sum_attn_lora_l32_1b_tc_en_v0 rp_attn_lora_l32_1b_tc_es_v0 cr_attn_lora_l32_1b_tc_generic_fr_v0 qa_attn_lora_l32_1b_tc_ja_v0 correction_attn_lora_l32_1b_tc_ko_v0"
experiments_list[3]="cr_attn_lora_l32_1b_tc_generic_en_v0 qa_attn_lora_l32_1b_tc_es_v0 correction_attn_lora_l32_1b_tc_fr_v0 sum_attn_lora_l32_1b_tc_ja_v0 rp_attn_lora_l32_1b_tc_zh_v0"
experiments_list[4]="correction_attn_lora_l32_1b_tc_en_v0 sum_attn_lora_l32_1b_tc_es_v0 rp_attn_lora_l32_1b_tc_it_v0 cr_attn_lora_l32_1b_tc_generic_ja_v0 qa_attn_lora_l32_1b_tc_zh_v0"
experiments_list[5]="rp_attn_lora_l32_1b_tc_de_v0 cr_attn_lora_l32_1b_tc_generic_es_v0 qa_attn_lora_l32_1b_tc_it_v0 correction_attn_lora_l32_1b_tc_ja_v0 sum_attn_lora_l32_1b_tc_zh_v0"
experiments_list[6]="qa_attn_lora_l32_1b_tc_de_v0 correction_attn_lora_l32_1b_tc_es_v0 sum_attn_lora_l32_1b_tc_it_v0 rp_attn_lora_l32_1b_tc_ko_v0 cr_attn_lora_l32_1b_tc_generic_zh_v0"
experiments_list[7]="sum_attn_lora_l32_1b_tc_de_v0 rp_attn_lora_l32_1b_tc_fr_v0 cr_attn_lora_l32_1b_tc_generic_it_v0 qa_attn_lora_l32_1b_tc_ko_v0 correction_attn_lora_l32_1b_tc_zh_v0"

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