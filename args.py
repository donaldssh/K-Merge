import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-3.2-1B-Instruct"
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    stored_model_name: str = field(
        default="", metadata={"help": "The name of the model that is stored"}
    )
    local_model_repo_path: str = field(
        default="stored_models",
        metadata={"help": "The path where model checkpoints will be store"},
    )
    lora_merge_modules: list[str] = field(
        default_factory=list,
        metadata={"help": "The name of the lora model that is stored"},
    )
    lora_merge_strategy: str = field(
        default="",
        metadata={"help": "The name of the strategy for merging lora parameters"},
    )
    attn_lora: bool = field(
        default=False, metadata={"help": "If to apply lora only to attn parameters."}
    )
    qv_lora: bool = field(
        default=False, metadata={"help": "If to apply lora only to qv parameters."}
    )
    initial_lora_module: str = field(
        default_factory=list,
        metadata={"help": "The name of the initial lora, for continual merging"},
    )
    shuffle_loras: bool = field(
        default=False, metadata={"help": "shuffle the lora_merge_modules"}
    )
    adversarial_order: bool = field(
        default=False,
        metadata={
            "help": "shuffle the lora_merge_modules forcing adversarial ordering"
        },
    )
    continual_eval_every_step: bool = field(
        default=False, metadata={"help": "eval at every incremental step"}
    )
    manual_cluster_by_task: bool = field(
        default=False,
        metadata={"help": "whether to manually cluster the loras by task"},
    )
    random_clustering: bool = field(
        default=False, metadata={"help": "whether to random cluster the loras"}
    )
    threshold_sim: float = field(
        default=0.0,
        metadata={"help": "active when != 0. threshold used to create a new cluster"},
    )
    similarity: str = field(
        default="cos",
        metadata={
            "choices": ["cos", "l2", "l1", "linf"],
            "help": "The similarity metric used for clustering LoRAs.",
        },
    )
    dec_peft: bool = field(default=False)


@dataclass
class DataArguments:
    dataset: str = field(
        default="text-correction", metadata={"help": "Which dataset to use."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=512,
        metadata={
            "help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    target_max_len: int = field(
        default=512,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    local_repo: str = field(
        default="./data",
        metadata={"help": "Local directory where datasets are stored."},
    )
    languages: str = field(
        default=None,
        metadata={
            "help": "For datasets that support multiple languages, the languages for which examples should be loaded"
        },
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train on the input in addition to the target text."
        },
    )
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    experiment_name: str = field(
        default="dbg", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="adamw_bnb_8bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=3,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={
            "help": "The evaluation batch size per GPU. Increase for better speed."
        },
    )
    seed: int = field(default=0, metadata={"help": "The general seed to use."})
    data_seed: int = field(
        default=42, metadata={"help": "The seed to use for sampling data."}
    )
    dataloader_num_workers: int = field(
        default=3, metadata={"help": "The number of workers for data loading."}
    )
    # use gradient_accumulation_steps of 4 for tone adj. due to its smaller size
    gradient_accumulation_steps: int = field(
        default=64,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    eval_accumulation_steps: int = field(
        default=4, metadata={"help": "How many items to accumulate during evaluation"}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "How many epochs to use for training the models"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    lr_scheduler_type: str = field(
        default="cosine", metadata={"help": "Learning rate schedule"}
    )
    warmup_steps: int = field(default=10, metadata={"help": "Number of warmup steps"})
    logging_steps: int = field(
        default=5000,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    save_strategy: str = field(
        default="no", metadata={"help": "When to save checkpoints"}
    )
    evaluation_strategy: str = field(
        default="epoch", metadata={"help": "When to evaluate models"}
    )
    save_total_limit: int = field(
        default=2,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    do_eval: bool = field(
        default=False, metadata={"help": "To val or not to val, that is the question?"}
    )
    do_valid: bool = field(
        default=False,
        metadata={
            "help": "If we do separate evaluation on the validation set in the same way as testing."
        },
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "To test or not to test, that is the question?"},
    )
    lora_r: int = field(default=32, metadata={"help": "The value of LoRA r parameter."})
    lora_alpha: int = field(
        default=16, metadata={"help": "The value of LoRA alpha parameter."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "The value of LoRA dropout parameter."}
    )
    padding_side: str = field(
        default="left",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    zero_shot: bool = field(
        default=False,
        metadata={
            "help": "If to do the special case of not using any LoRA parameters."
        },
    )
    predict_with_generate: bool = field(
        default=False,
        metadata={
            "help": "If we do separate evaluation on the validation set in the same way as testing."
        },
    )
    tone: str = field(
        default="professional",
        metadata={
            "help": "Selected tone for the tone task. Options: casual, professional, witty, paraphrase, generic."
        },
    )
    dataset_language: str = field(
        default="en", metadata={"help": "Language of the task"}
    )
    src_lang: str = field(
        default="en", metadata={"help": "Target language of the translation task"}
    )
    tgt_lang: str = field(
        default="es", metadata={"help": "Target language of the translation task"}
    )
    num_examples_merge: int = field(
        default=5, metadata={"help": "Number of examples used for merging."}
    )
    density: float = field(
        default=0.5, metadata={"help": "What fraction of values to prune in merging"}
    )
    max_num_tokens_eval: int = field(
        default=128,
        metadata={
            "help": "If to manually set the number of new tokens for evaluation."
        },
    )
    opcm_alpha: float = field(default=1.0, metadata={"help": "opcm alpha"})

    use_all_models_at_step: bool = field(
        default=False,
        metadata={
            "help": "wheter to use all available loras at each merging step, sort of a upper bound"
        },
    )

    k_lora_budget: int = field(
        default=5, metadata={"help": "Max number of LoRAs that could be stored"}
    )


@dataclass
class GenerationArguments:
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=32,
        metadata={
            "help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
            "if predict_with_generate is set."
        },
    )
    min_new_tokens: Optional[int] = field(
        default=None, metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def get_arguments():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    model_args, data_args, training_args, generation_args, _ = (
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    )
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args)
    )
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    training_args.output_dir = args.output_dir
    print(args)

    return args, training_args
