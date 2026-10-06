import copy
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset, DatasetDict
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    pipeline,
    set_seed,
)
from typing_extensions import TypeAlias

import data_loading
from args import get_arguments
from merge import merge_loras_cluster
from utils import (
    compute_task_metrics,
    get_task_prompt,
    print_trainable_parameters,
    subsample_dataset,
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
import random

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

ALL_TONES = ["professional", "casual", "witty", "paraphrase"]


def get_model(args, quantization_config, torch_dtype=torch.float16):
    # function to load the model and the tokenizer

    if args.do_train:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        # specify where to apply the lora parameters
        if args.attn_lora:
            lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif args.qv_lora:
            lora_target_modules = ["q_proj", "v_proj"]
        else:
            lora_target_modules = [
                "gate_proj",
                "down_proj",
                "up_proj",
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
            ]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_target_modules,
        )

        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
    elif args.zero_shot:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            quantization_config=quantization_config,
        )
    else:
        stored_model_path = os.path.join(
            args.local_model_repo_path, args.stored_model_name
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
            stored_model_path,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            adapter_name=args.stored_model_name,
            quantization_config=quantization_config,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side=args.padding_side,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        model_max_length=4096,
    )

    tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, "Padding token should not be empty"

    return model, tokenizer


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}" for example in instances
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}" for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    # function to load and format the data - format is {`input`, `output`}

    def add_prompt(dataset, task_prompt, split, tone):
        # add prompt to the dataset, either as a separate system prompt or within the user prompt
        if tone == "generic":
            num_samples_per_tone = len(dataset[split]["input"]) // len(ALL_TONES)

        # training split also includes the targets
        formatted_data = []
        if split == "train":
            for sample_index, (input_text, target_text) in enumerate(
                zip(dataset[split]["input"], dataset[split]["output"])
            ):
                if tone == "generic" and args.dataset == "content_rephrasing":
                    content = (
                        task_prompt[sample_index // num_samples_per_tone] + input_text
                    )
                else:
                    content = task_prompt + input_text
                formatted_data.append(
                    [
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": target_text},
                    ]
                )
        else:
            for sample_index, text in enumerate(dataset[split]["input"]):
                if tone == "generic" and args.dataset == "content_rephrasing":
                    content = task_prompt[sample_index // num_samples_per_tone] + text
                else:
                    content = task_prompt + text
                formatted_data.append([{"role": "user", "content": content}])

        return formatted_data

    def format_dataset(dataset, dataset_name, dataset_language, tone):
        # format the dataset, e.g. adding the task prompt

        # add task prompt to the data
        task_prompt = get_task_prompt(dataset_name, dataset_language, tone)

        data_train = add_prompt(dataset, task_prompt, "train", tone)
        data_val = add_prompt(dataset, task_prompt, "validation", tone)
        data_test = add_prompt(dataset, task_prompt, "test", tone)

        dataset["train"] = Dataset.from_dict({"output": data_train}).map(
            lambda x: {
                "input": "",
                "output": tokenizer.apply_chat_template(
                    x["output"], tokenize=False, add_generation_prompt=False
                ),
            }
        )
        dataset["validation"] = Dataset.from_dict(
            {"input": data_val, "output": dataset["validation"]["output"]}
        )
        dataset["test"] = Dataset.from_dict(
            {"input": data_test, "output": dataset["test"]["output"]}
        )

        # Remove unused columns
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["input", "output"]
            ]
        )
        return dataset

    def load_task_data(args):
        # load and format the data for the given task

        train_dataset, eval_dataset, test_dataset, merge_dataset = (
            None,
            None,
            None,
            None,
        )
        dataset = data_loading.load_own_data(
            dataset_spec=args.dataset,
            local_repo=args.local_repo,
            tone=args.tone,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            dataset_language=args.dataset_language,
        )

        datasets_dict = {
            "train": {"input": [], "output": []},
            "validation": {"input": [], "output": []},
            "test": {"input": [], "output": []},
            "merge": {"input": [], "output": []},
        }

        dataset = format_dataset(
            dataset, args.dataset, args.dataset_language, args.tone
        )

        if args.do_eval or args.do_valid:
            datasets_dict["validation"] = subsample_dataset(
                dataset, "validation", args.max_eval_samples
            )
        if args.do_predict:
            datasets_dict["test"] = subsample_dataset(
                dataset, "test", args.max_test_samples
            )
        if args.lora_merge_modules:
            datasets_dict["merge"] = subsample_dataset(
                dataset, "train", args.num_examples_merge
            )
        if args.do_train:
            datasets_dict["train"] = subsample_dataset(
                dataset, "train", args.max_train_samples
            )

        dataset = DatasetDict(datasets_dict)

        if args.do_eval or args.do_valid:
            eval_dataset = dataset["validation"]
        if args.do_predict:
            test_dataset = dataset["test"]
        if args.do_train:
            train_dataset = dataset["train"]
        if args.lora_merge_modules:
            merge_dataset = dataset["merge"]

        return train_dataset, eval_dataset, test_dataset, merge_dataset

    train_dataset, eval_dataset, test_dataset, merge_dataset = load_task_data(args)
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval or args.do_valid else None,
        test_dataset=test_dataset if args.do_predict else None,
        merge_dataset=merge_dataset if args.lora_merge_modules else None,
        data_collator=data_collator,
    )


def make_data_module_incremental(
    tokenizer: transformers.PreTrainedTokenizer, args, incremental_step, dataset_name
) -> Dict:
    # function to load and format the data - format is {`input`, `output`}

    def add_prompt(dataset, task_prompt, split, tone, dataset_name=None):
        # add prompt to the dataset, either as a separate system prompt or within the user prompt
        if tone == "generic":
            num_samples_per_tone = len(dataset[split]["input"]) // len(ALL_TONES)

        formatted_data = []
        for sample_index, text in enumerate(dataset[split]["input"]):
            if tone == "generic" and dataset_name == "content_rephrasing":
                content = task_prompt[sample_index // num_samples_per_tone] + text
            else:
                content = task_prompt + text
            formatted_data.append([{"role": "user", "content": content}])

        return formatted_data

    def format_dataset(dataset, dataset_name, dataset_language, tone):
        # format the dataset, e.g. adding the task prompt
        # add task prompt to the data
        task_prompt = get_task_prompt(dataset_name, dataset_language, tone)

        data_val = add_prompt(
            dataset, task_prompt, "validation", tone, dataset_name=dataset_name
        )
        data_test = add_prompt(
            dataset, task_prompt, "test", tone, dataset_name=dataset_name
        )

        dataset["validation"] = Dataset.from_dict(
            {"input": data_val, "output": dataset["validation"]["output"]}
        )
        dataset["test"] = Dataset.from_dict(
            {"input": data_test, "output": dataset["test"]["output"]}
        )
        return dataset

    def load_task_data(args):
        # load and format the data for the given task

        eval_dataset, test_dataset = None, None
        dataset = data_loading.load_own_data(
            dataset_spec=dataset_name,
            local_repo=args.local_repo,
            tone=args.tone,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            dataset_language=args.lora_merge_modules[incremental_step][-6:-4],
        )

        datasets_dict = {
            "validation": {"input": [], "output": []},
            "test": {"input": [], "output": []},
        }

        if ("persona-chat-synthetic" in dataset_name) or ("squad" in dataset_name):
            if not args.max_eval_samples:
                max_eval_samples = 1000
            else:
                max_eval_samples = args.max_eval_samples
            if not args.max_test_samples:
                max_test_samples = 1000
            else:
                max_test_samples = args.max_test_samples
        else:
            max_eval_samples = args.max_eval_samples
            max_test_samples = args.max_test_samples

        if args.do_eval or args.do_valid:
            datasets_dict["validation"] = subsample_dataset(
                dataset, "validation", max_eval_samples
            )
        if args.do_predict:
            datasets_dict["test"] = subsample_dataset(dataset, "test", max_test_samples)

        dataset = format_dataset(
            DatasetDict(datasets_dict),
            dataset_name=dataset_name,
            dataset_language=args.lora_merge_modules[incremental_step][-6:-4],
            tone=args.tone,
        )

        if args.do_eval or args.do_valid:
            eval_dataset = dataset["validation"]
        if args.do_predict:
            test_dataset = dataset["test"]

        return eval_dataset, test_dataset

    eval_dataset, test_dataset = load_task_data(args)
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        eval_dataset=eval_dataset if args.do_eval or args.do_valid else None,
        test_dataset=test_dataset if args.do_predict else None,
        data_collator=data_collator,
    )


def evaluation_loop(
    args,
    trainer,
    stage_keywords,
    stage,
    data_module,
    test_pipe,
    terminators,
    dataset_name=None,
):
    # function to run an evaluation loop for the given split

    # ensure we always perform the evaluation with the same seed to ensure consistency across methods
    set_seed(args.seed)
    logger.info(f"*** {stage_keywords[stage][0]} ***")
    metrics = trainer.evaluate(
        eval_dataset=data_module[stage_keywords[stage][1]],
        metric_key_prefix=stage_keywords[stage][2],
    )
    trainer.log_metrics(stage_keywords[stage][2], metrics)
    trainer.save_metrics(stage_keywords[stage][2], metrics)

    outputs = []
    references = [e.strip() for e in data_module[stage_keywords[stage][1]]["output"]]

    start_time = time.time()
    formatted_input = data_module[stage_keywords[stage][1]]["input"]

    # generate outputs from the evaluated model
    for generated_texts in test_pipe(
        formatted_input,
        batch_size=args.per_device_eval_batch_size,
        truncation="only_first",
        return_full_text=False,
        do_sample=False,
        num_return_sequences=1,
        top_p=1.0,
        eos_token_id=terminators,
    ):
        outputs.append([e["generated_text"].strip() for e in generated_texts])

    eval_time = time.time() - start_time
    inputs = [
        e[-1]["content"].strip() for e in data_module[stage_keywords[stage][1]]["input"]
    ]

    if dataset_name is not None:
        dataset = dataset_name[:-3]
        language = dataset_name[-2:]
    else:
        dataset = args.dataset
        language = args.dataset_language

    scores = compute_task_metrics(references, outputs, inputs, dataset, language)
    scores = {f"{stage_keywords[stage][2]}_{k}": v for k, v in scores.items()}
    print(stage_keywords[stage][0] + " scores: ", scores)
    trainer.save_metrics(stage_keywords[stage][2], scores)
    trainer.save_metrics(
        stage_keywords[stage][2],
        {
            f"{stage_keywords[stage][2]}_eval_time": eval_time,
            f"{stage_keywords[stage][2]}_time_per_example": eval_time
            / scores[f"{stage_keywords[stage][2]}_n_examples"],
        },
    )
    examples = {
        f"{stage_keywords[stage][2]}_inputs": inputs[:3],
        f"{stage_keywords[stage][2]}_targets": references[:3],
        f"{stage_keywords[stage][2]}_outputs": outputs[:3],
    }

    trainer.save_metrics(stage_keywords[stage][2], examples)

    return references, outputs, inputs


def run_evaluation_continual(
    args, trainer, tokenizer, data_module, incremental_step, dataset_name_lang
):
    # function to run the overall evaluation

    if args.do_valid or args.do_predict:
        if args.model_name_or_path == "meta-llama/Llama-3.2-1B-Instruct":
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                tokenizer.convert_tokens_to_ids("<|eom_id|>"),
                tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            ]
        else:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            ]
        test_pipe = pipeline(
            task="text-generation",
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            max_new_tokens=args.max_num_tokens_eval,
        )

    stage_keywords = {
        "val": [
            "Validation",
            "eval_dataset",
            f"valid_{dataset_name_lang}_{incremental_step}",
        ],
        "test": [
            "Test",
            "test_dataset",
            f"test_{dataset_name_lang}_{incremental_step}",
        ],
    }
    stages = []
    if args.do_valid:
        stages.append("val")
    if args.do_predict:
        stages.append("test")

    # we will store the predictions so that we can later assess them using llm judge
    generated_predictions = {}
    for stage in stages:
        references, outputs, inputs = evaluation_loop(
            args,
            trainer,
            stage_keywords,
            stage,
            data_module,
            test_pipe,
            terminators,
            dataset_name=dataset_name_lang,
        )
        generated_predictions[stage] = {
            "references": references,
            "outputs": outputs,
            "inputs": inputs,
            "dataset": dataset_name_lang,
        }

    os.makedirs("predictions_continual/", exist_ok=True)
    with open("predictions_continual/" + args.experiment_name + ".json", "w") as f:
        json.dump(generated_predictions, f)


def run_evaluation(args, trainer, tokenizer, data_module):
    # function to run the overall evaluation

    if args.do_valid or args.do_predict:
        if args.model_name_or_path in ["meta-llama/Llama-3.2-1B-Instruct"]:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                tokenizer.convert_tokens_to_ids("<|eom_id|>"),
                tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            ]
        else:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            ]
        test_pipe = pipeline(
            task="text-generation",
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            max_new_tokens=args.max_num_tokens_eval,
        )

    stage_keywords = {
        "val": ["Validation", "eval_dataset", f"valid"],
        "test": ["Test", "test_dataset", f"test"],
    }
    stages = []
    if args.do_valid:
        stages.append("val")
    if args.do_predict:
        stages.append("test")

    # we will store the predictions so that we can later assess them using llm judge
    generated_predictions = {}
    for stage in stages:
        references, outputs, inputs = evaluation_loop(
            args,
            trainer,
            stage_keywords,
            stage,
            data_module,
            test_pipe,
            terminators,
        )
        generated_predictions[stage] = {
            "references": references,
            "outputs": outputs,
            "inputs": inputs,
            "dataset": args.dataset,
        }

    with open("predictions/" + args.experiment_name + ".json", "w") as f:
        json.dump(generated_predictions, f)


def compute_further_stats(trainer, args):
    # function to compute additional useful statistics

    num_params_main = 0
    num_params_loras = 0
    for name, param in trainer.model.named_parameters():
        if "lora" in name:
            num_params_loras += param.numel()
        else:
            num_params_main += param.numel()

    trainer.save_metrics(
        "overall",
        {
            "num_params_main": num_params_main,
            "num_params_loras": num_params_loras,
            "num_loras": len(args.lora_merge_modules),
            "gpu_memory": torch.cuda.memory_allocated(),
        },
    )


StateDictType: TypeAlias = Dict[str, Tensor]


# https://github.com/tanganke/fusion_bench/blob/b88aacb9f7d1f607b1ea7196da3b7829995f3337/fusion_bench/utils/parameters.py#L47
def state_dict_to_vector(
    state_dict: Union[StateDictType, nn.Module],
    remove_keys: Optional[List[str]] = None,
):
    """
    Convert a state dictionary to a vector.

    Args:
        state_dict (Union[dict[str, torch.Tensor], nn.Module]): The state dictionary to convert.
        remove_keys (list, optional): List of keys to remove from the state dictionary. Defaults to [].

    Returns:
        torch.Tensor: The converted vector.
    """
    remove_keys = remove_keys if remove_keys is not None else []

    if isinstance(state_dict, nn.Module):
        shared_state_dict = state_dict.state_dict()
    else:
        shared_state_dict = copy.copy(state_dict)

    # remove the keys to be removed
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]

    # sort the reference dict
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))

    vector = nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )
    return vector


def state_dict_sub(
    a: StateDictType, b: StateDictType, strict: bool = True, device=None
):
    """
    Returns the difference between two state dicts `a-b`.

    Args:
        a (StateDictType): The first state dict.
        b (StateDictType): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        StateDictType: The difference between the two state dicts.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = OrderedDict()
    for k in a:
        if k in b:
            diff[k] = a[k] - b[k]
            if device is not None:
                diff[k] = diff[k].to(device, non_blocking=True)
    return diff


def _svd(w: Tensor, full_matrices=True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform Singular Value Decomposition (SVD) on a tensor.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


def svd(
    w: Tensor, full_matrices=True, accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform SVD on a tensor, optionally using a specified accelerator.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.
        accelerator (str): The device to perform the computation on.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u.to(original_device), s.to(original_device), v.to(original_device)


def get_task_vector_norm(model, pretrained_model) -> Tensor:
    """
    Get the vector norm of the task model.

    Args:
        model (nn.Module): The task model.
        pretrained_model (nn.Module): The pretrained model.

    Returns:
        Tensor: The vector norm of the task model.
    """
    return torch.linalg.norm(
        state_dict_to_vector(state_dict_sub(model, pretrained_model))
    )


def merge_linear_weights(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    alpha: float,
    previous_lambda_t,
    lambda_t,
    accelerator: str = "cpu",
):
    original_device = merged_W.device
    merged_W = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv = task_W - pretrained_W

    u, s, v = svd(previous_merged_tv)
    split_rank = (s.cumsum(dim=0) / s.sum() > alpha).float().argmax().item()

    projected_task_tv = u.T @ task_tv @ v
    projected_task_tv.diag().fill_(0)

    projected_task_tv[:split_rank, :split_rank] = 0

    cleaned_task_tv = u @ projected_task_tv @ v.T

    new_merged_W = (
        pretrained_W
        + (previous_lambda_t * previous_merged_tv + cleaned_task_tv) / lambda_t
    )
    return new_merged_W.to(original_device)


def merge_other_parameters(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    previous_lambda_t,
    lambda_t,
    accelerator: str = "cpu",
):
    original_device = merged_W.device
    merged_W = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv = task_W - pretrained_W

    new_merged_W = (
        pretrained_W + (previous_lambda_t * previous_merged_tv + task_tv) / lambda_t
    )
    return new_merged_W.to(original_device)


def compute_pair_similarity(a: Tensor, b: Tensor, similarity_metric: str) -> float:
    """Similarity between two flattened tensors, per args.similarity ("cos", "l2", "l1", "linf")."""
    if similarity_metric == "cos":
        return F.cosine_similarity(a, b, dim=0).item()
    elif similarity_metric == "l2":
        # L2 (Frobenius) distance
        dist = torch.norm(a - b, p=2).item()
    elif similarity_metric == "l1":
        # L1 distance
        dist = torch.norm(a - b, p=1).item()
    elif similarity_metric == "linf":
        # L inf norm
        dist = torch.max(torch.abs(a - b)).item()
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    return 1 / (1 + dist)


def run_experiment():
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    args, training_args = get_arguments()

    quantization_config = None
    torch_dtype = torch.float16

    model, tokenizer = get_model(args, quantization_config, torch_dtype)

    model.config.use_cache = False
    print_trainable_parameters(model)
    set_seed(args.seed)

    # Synthetic Persona Chat dataset is large so we only use a subset of the examples in our evaluation
    if args.dataset == "persona-chat-synthetic" or args.dataset == "squad":
        print(
            "Using 1000 samples for each of validation and testing for Synthetic Persona Chat dataset"
        )
        # we use subset of samples for val and test for reply - always the same set because of random seed
        if not args.max_eval_samples:
            args.max_eval_samples = 1000
        if not args.max_test_samples:
            args.max_test_samples = 1000

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{
            k: v
            for k, v in data_module.items()
            if k != "test_dataset" and k != "merge_dataset"
        },
    )

    if args.do_train:
        start_time = time.time()
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        stored_model_path = os.path.join(
            args.local_model_repo_path, args.stored_model_name
        )
        trainer.model.save_pretrained(stored_model_path)

        eval_loss_per_epoch = {
            "eval_loss_per_epoch": [
                e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e
            ]
        }
        trainer.save_metrics("train", eval_loss_per_epoch)
        trainer.save_metrics("train", {"training_time": time.time() - start_time})

    if not args.lora_merge_modules:
        # Plain single-task train/zero-shot run: nothing to merge, so skip the
        # continual-merging setup below
        run_evaluation(args, trainer, tokenizer, data_module)
        return

    if not args.zero_shot and args.initial_lora_module:
        _ = model.load_adapter(
            os.path.join(args.local_model_repo_path, args.initial_lora_module),
            adapter_name="merge_0",
        )
    elif not args.zero_shot:

        if args.shuffle_loras:
            random.shuffle(args.lora_merge_modules)
        elif args.adversarial_order:
            base_path = args.lora_merge_modules[0].split("/")[0]
            random_task_order = ["correction", "cr", "qa", "rp", "sum"]
            random_lang_order = ["en", "de", "es", "fr", "it", "ja", "ko", "zh"]

            random.shuffle(random_task_order)
            random.shuffle(random_lang_order)

            if args.model_name_or_path == "meta-llama/Llama-3.2-1B-Instruct":
                lora_tag = "l32_1b_tc"
            elif args.model_name_or_path in ["Qwen/Qwen2.5-1.5B-Instruct"]:
                lora_tag = "qwen2515b_ob"
            else:
                raise ValueError(
                    f"Unknown model for adversarial_order lora naming: {args.model_name_or_path}"
                )

            new_lora_order = []
            for task in random_task_order:
                for lang in random_lang_order:
                    if task == "cr":
                        selected_lora = os.path.join(
                            base_path, f"{task}_attn_lora_{lora_tag}_generic_{lang}_v0/"
                        )
                    else:
                        selected_lora = os.path.join(
                            base_path, f"{task}_attn_lora_{lora_tag}_{lang}_v0/"
                        )

                    new_lora_order.append(selected_lora)

            args.lora_merge_modules = new_lora_order
            print(f"TASK ORDER {random_task_order}")
            print(f"LANG ORDER {random_lang_order}")

        for lora_name in args.lora_merge_modules:
            print(
                f"task: {lora_name.split('/')[1].split('_attn')[0]} , lang: {lora_name[-6:-4]}"
            )

        initial_lora_module = args.lora_merge_modules[0]

        mapping_clusters = {k: [] for k in range(args.k_lora_budget)}
        mapping_clusters[0].append(initial_lora_module)
        initial_adapter_name = "cluster_0"

        _ = model.load_adapter(
            os.path.join(args.local_model_repo_path, initial_lora_module),
            adapter_name=initial_adapter_name,
        )

    else:
        print("zero shot start, not loading lora model")

    data_module_dict = {}

    dataset_mapping = {
        "correction": "text-correction",
        "qa": "squad",
        "rp": "persona-chat-synthetic",
        "sum": "samsum",
        "cr": "content_rephrasing",
    }

    for incremental_step in range(
        len(args.lora_merge_modules)
    ):  # "lora_path_1,lora_path_2,...,lora_path_n"
        dataset_name_str = (
            args.lora_merge_modules[incremental_step].split("/")[1].split("_")[0]
        )
        lang = args.lora_merge_modules[incremental_step][-6:-4]
        dataset_name = dataset_mapping[dataset_name_str]
        if lang != "en" and dataset_name == "text-correction":
            dataset_name = "text-correction-org-lang"
        data_module_i = make_data_module_incremental(
            tokenizer=tokenizer,
            args=args,
            incremental_step=incremental_step,
            dataset_name=dataset_name,
        )
        data_module_dict[dataset_name + f"_{lang}"] = data_module_i

    if args.zero_shot:
        for task_name_lang, data_module_v in data_module_dict.items():
            print(f"eval on {task_name_lang}")
            run_evaluation_continual(
                args,
                trainer,
                tokenizer,
                data_module_v,
                incremental_step=0,
                dataset_name_lang=task_name_lang,
            )

    def get_lora_deltas(model, adapter_name):
        deltas = {}
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                if adapter_name in module.lora_A and adapter_name in module.lora_B:
                    lora_A = module.lora_A[adapter_name].weight.detach()
                    lora_B = module.lora_B[adapter_name].weight.detach()
                    # Effective weight update
                    delta_W = torch.matmul(lora_B, lora_A)
                    deltas[name] = delta_W
        return deltas

    if args.lora_merge_strategy == "opcm":
        merged_model_opcm = {}
        pretrained_model_opcm = {}
        all_model = model.base_model.state_dict()

        lora_only_params = {k: v for k, v in all_model.items() if "cluster_0" in k}
        pretrained_model_opcm[0] = {
            k: torch.tensor(0) for k, _ in lora_only_params.items()
        }
        merged_model_opcm[0] = lora_only_params

        previous_lambda_t = {k: 1 for k in range(args.k_lora_budget)}
        avg_task_vector_norm = get_task_vector_norm(
            merged_model_opcm[0], pretrained_model_opcm[0]
        ).item()

        all_task_vector_norm = {}
        all_task_vector_norm[0] = [avg_task_vector_norm]

        lambda_t = None

    clustering_time_list = []
    for incremental_step, incremental_lora in enumerate(
        args.lora_merge_modules
    ):  # "lora_path_1,lora_path_2,...,lora_path_n"
        if incremental_step == 0:
            continue

        if args.lora_merge_strategy == "opcm":

            if incremental_step < args.k_lora_budget:
                print(f"adding lora {incremental_step}: {incremental_lora}")
                _ = model.load_adapter(
                    os.path.join(args.local_model_repo_path, incremental_lora),
                    adapter_name=f"cluster_{incremental_step}",
                )
                mapping_clusters[incremental_step].append(incremental_lora)

                all_model = model.base_model.state_dict()
                lora_only_params = {
                    k: v
                    for k, v in all_model.items()
                    if f"cluster_{incremental_step}" in k
                }
                pretrained_model_opcm[incremental_step] = {
                    k: torch.tensor(0) for k, _ in lora_only_params.items()
                }
                merged_model_opcm[incremental_step] = lora_only_params

                previous_lambda_t = {k: 1 for k in range(args.k_lora_budget)}
                avg_task_vector_norm = get_task_vector_norm(
                    merged_model_opcm[incremental_step],
                    pretrained_model_opcm[incremental_step],
                ).item()

                all_task_vector_norm[incremental_step] = [avg_task_vector_norm]

                lambda_t = None

            else:
                if "new_lora" in model.peft_config.keys():
                    model.delete_adapter("new_lora")
                _ = model.load_adapter(
                    os.path.join(args.local_model_repo_path, incremental_lora),
                    adapter_name="new_lora",
                )

                new_weights = get_lora_deltas(model, "new_lora")
                max_similarity = float("-inf")
                closest_cluster = None
                for cluster_id in range(args.k_lora_budget):
                    adapter_name = f"cluster_{cluster_id}"
                    cluster_weights = get_lora_deltas(model, adapter_name)

                    similarity = 0.0
                    count = 0
                    for key in new_weights:
                        if key in cluster_weights:
                            # Flatten both tensors
                            a = new_weights[key].flatten()
                            b = cluster_weights[key].flatten()

                            # Cosine similarity
                            sim = compute_pair_similarity(a, b, args.similarity)
                            similarity += sim
                            count += 1

                    if count > 0:
                        similarity /= count  # Average similarity across layers

                    if similarity > max_similarity:
                        max_similarity = similarity
                        closest_cluster = cluster_id

                mapping_clusters[closest_cluster].append(incremental_lora)
                cluster_step = len(mapping_clusters[closest_cluster])
                print(
                    f"CONTINUAL OPCM: merging with {incremental_lora} cluster {closest_cluster}"
                )

                all_model = model.base_model.state_dict()
                task_model = {
                    k.replace(incremental_lora, f"cluster_{closest_cluster}"): v
                    for k, v in all_model.items()
                    if incremental_lora in k
                }

                all_task_vector_norm[closest_cluster].append(
                    get_task_vector_norm(
                        task_model, pretrained_model_opcm[closest_cluster]
                    ).item()
                )
                avg_task_vector_norm = np.mean(all_task_vector_norm[closest_cluster])
                lambda_t = 1  # temporary value

                temp_model = copy.deepcopy(merged_model_opcm[closest_cluster])

                for module_name, module in merged_model_opcm[closest_cluster].items():

                    if module_name.split(".")[-1] == "weight":
                        temp_model[module_name] = merge_linear_weights(
                            module,
                            pretrained_model_opcm[closest_cluster][module_name],
                            task_model[module_name],
                            alpha=args.opcm_alpha,
                            previous_lambda_t=previous_lambda_t[closest_cluster],
                            lambda_t=lambda_t,
                            accelerator=model.device,
                        )
                    else:
                        temp_model[module_name] = merge_other_parameters(
                            module,
                            pretrained_model_opcm[closest_cluster][module_name],
                            task_model[module_name],
                            previous_lambda_t=previous_lambda_t[closest_cluster],
                            lambda_t=lambda_t,
                            accelerator=model.device,
                        )

                new_merged_model = copy.deepcopy(merged_model_opcm[closest_cluster])

                task_vector_norm = get_task_vector_norm(
                    merged_model_opcm[closest_cluster],
                    pretrained_model_opcm[closest_cluster],
                )
                lambda_t *= task_vector_norm / avg_task_vector_norm

                for param_name, param in temp_model.items():
                    new_merged_model[param_name] = pretrained_model_opcm[
                        closest_cluster
                    ][param_name] + (
                        param - pretrained_model_opcm[closest_cluster][param_name]
                    ) * (
                        avg_task_vector_norm / task_vector_norm
                    )

                previous_lambda_t[closest_cluster] = lambda_t
                lambda_t = None

                merged_model_opcm[closest_cluster] = new_merged_model
                model.base_model.load_state_dict(
                    merged_model_opcm[closest_cluster], strict=False
                )

        else:

            if args.use_all_models_at_step:
                _ = model.load_adapter(
                    os.path.join(args.local_model_repo_path, incremental_lora),
                    adapter_name=incremental_lora,
                )

            if args.threshold_sim != 0:
                if "new_lora" in model.peft_config.keys():
                    model.delete_adapter("new_lora")
                _ = model.load_adapter(
                    os.path.join(args.local_model_repo_path, incremental_lora),
                    adapter_name="new_lora",
                )

                new_weights = get_lora_deltas(model, "new_lora")
                diff = 0
                count_diff = 0

                not_empty = sum(1 for v in mapping_clusters.values() if v != [])
                if not_empty < args.k_lora_budget:

                    for cluster_id in (
                        k for k, v in mapping_clusters.items() if v != []
                    ):
                        adapter_name = f"cluster_{cluster_id}"
                        cluster_weights = get_lora_deltas(model, adapter_name)

                        similarity = 0.0
                        count = 0
                        for key in new_weights:
                            if key in cluster_weights:
                                a = new_weights[key].flatten()
                                b = cluster_weights[key].flatten()

                                sim = compute_pair_similarity(a, b, args.similarity)
                                similarity += sim
                                count += 1

                        if count > 0:
                            similarity /= count  # Average similarity across layers

                        if similarity < args.threshold_sim:
                            diff += 1

                        print(f"similarity {similarity}")
                        count_diff += 1

                # create a new cluster if the lora is different (more than a threshold) to all existing clusters
                if diff == count_diff and not_empty < args.k_lora_budget:
                    print(f"adding lora {incremental_lora} to cluster {cluster_id+1}")
                    _ = model.load_adapter(
                        os.path.join(args.local_model_repo_path, incremental_lora),
                        adapter_name=f"cluster_{cluster_id+1}",
                    )
                    mapping_clusters[cluster_id + 1].append(incremental_lora)
                else:
                    new_weights = get_lora_deltas(model, "new_lora")
                    max_similarity = float("-inf")
                    closest_cluster = None
                    for cluster_id in range(args.k_lora_budget):
                        adapter_name = f"cluster_{cluster_id}"
                        cluster_weights = get_lora_deltas(model, adapter_name)

                        similarity = 0.0
                        count = 0
                        for key in new_weights:
                            if key in cluster_weights:
                                # Flatten both tensors
                                a = new_weights[key].flatten()
                                b = cluster_weights[key].flatten()

                                sim = compute_pair_similarity(a, b, args.similarity)
                                similarity += sim
                                count += 1

                        if count > 0:
                            similarity /= count  # Average similarity across layers

                        if similarity > max_similarity:
                            max_similarity = similarity
                            closest_cluster = cluster_id

                    mapping_clusters[closest_cluster].append(incremental_lora)
                    cluster_step = len(mapping_clusters[closest_cluster])
                    print(f"merging with {incremental_lora} cluster {closest_cluster}")
                    merge_loras_cluster(
                        trainer.model,
                        args,
                        incremental_step,
                        cluster_step,
                        closest_cluster,
                    )

            elif incremental_step < args.k_lora_budget:
                print(f"adding lora {incremental_step}: {incremental_lora}")
                _ = model.load_adapter(
                    os.path.join(args.local_model_repo_path, incremental_lora),
                    adapter_name=f"cluster_{incremental_step}",
                )
                mapping_clusters[incremental_step].append(incremental_lora)

            else:
                if "new_lora" in model.peft_config.keys():
                    model.delete_adapter("new_lora")
                _ = model.load_adapter(
                    os.path.join(args.local_model_repo_path, incremental_lora),
                    adapter_name="new_lora",
                )

                clustering_time = time.time()

                new_weights = get_lora_deltas(model, "new_lora")
                max_similarity = float("-inf")
                closest_cluster = None
                for cluster_id in range(args.k_lora_budget):
                    adapter_name = f"cluster_{cluster_id}"
                    cluster_weights = get_lora_deltas(model, adapter_name)

                    similarity = 0.0
                    count = 0
                    for key in new_weights:
                        if key in cluster_weights:
                            # Flatten both tensors
                            a = new_weights[key].flatten()
                            b = cluster_weights[key].flatten()

                            sim = compute_pair_similarity(a, b, args.similarity)
                            similarity += sim
                            count += 1

                    if count > 0:
                        similarity /= count  # Average similarity across layers

                    if similarity > max_similarity:
                        max_similarity = similarity
                        closest_cluster = cluster_id

                end_clustering_time = time.time() - clustering_time
                clustering_time_list.append(end_clustering_time)

                if args.random_clustering:
                    print(f"closest cluster id: {closest_cluster}")
                    closest_cluster = np.random.randint(args.k_lora_budget)
                    print(f"randomly assigned to: {closest_cluster}")
                if args.manual_cluster_by_task:
                    task_new_lora = incremental_lora.split("/")[1].split("_")[0]
                    for k_id in range(args.k_lora_budget):
                        if (
                            mapping_clusters[k_id][0].split("/")[1].split("_")[0]
                            == task_new_lora
                        ):
                            closest_cluster = k_id
                            break

                mapping_clusters[closest_cluster].append(incremental_lora)
                cluster_step = len(mapping_clusters[closest_cluster])
                print(f"merging with {incremental_lora} cluster {closest_cluster}")
                merge_loras_cluster(
                    trainer.model, args, incremental_step, cluster_step, closest_cluster
                )

        if args.continual_eval_every_step:
            for k, data_module_v in data_module_dict.items():
                run_evaluation_continual(
                    args,
                    trainer,
                    tokenizer,
                    data_module_v,
                    incremental_step=incremental_step + 1,
                    dataset_name_lang=k,
                )

    print(mapping_clusters)
    print(f"TIMES: {clustering_time_list}")

    for task_name_lang, data_module_v in data_module_dict.items():
        print(f"eval on {task_name_lang}")

        cluster_id = -1
        for c_id, values in mapping_clusters.items():
            for v in values:
                task_name_key = v.split("/")[1].split("_attn")[0]

                if "generic" in task_name_key:
                    task_name_key = task_name_key.split("_generic")[0]
                lang = v[-6:-4]
                task_name = dataset_mapping[task_name_key]
                if lang != "en" and task_name == "text-correction":
                    task_name = "text-correction-org-lang"
                task_name += "_" + lang
                if task_name == task_name_lang:
                    cluster_id = c_id
                    break
            if cluster_id != -1:
                break

        print(f"task {task_name_lang} cluster {cluster_id}")
        model.set_adapter(f"cluster_{cluster_id}")

        run_evaluation_continual(
            args,
            trainer,
            tokenizer,
            data_module_v,
            incremental_step=incremental_step + 1,
            dataset_name_lang=task_name_lang,
        )

    compute_further_stats(trainer, args)


if __name__ == "__main__":
    run_experiment()
    print("end exp")
