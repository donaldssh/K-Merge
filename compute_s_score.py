import argparse
import json
import os

test_metrics = {
    "text-correction": "f05",
    "squad": "f1",
    "persona-chat-synthetic": "weighted_rouge",
    "content_rephrasing": "rougeL",
    "samsum": "rougeL",
}

metrics_order_to_print = [
    "text-correction",
    "squad",
    "persona-chat-synthetic",
    "samsum",
    "content_rephrasing",
]


single_lora_scores_llama = {
    "text-correction": {
        "en": 35.1,
        "de": 26.0,
        "es": 34.0,
        "fr": 20.5,
        "it": 27.2,
        "ja": 12.9,
        "ko": 9.2,
        "zh": 37.9,
    },
    "squad": {
        "en": 63.5,
        "de": 36.5,
        "es": 42.3,
        "fr": 34.7,
        "it": 37.1,
        "ja": 22.5,
        "ko": 28.2,
        "zh": 22.6,
    },
    "persona-chat-synthetic": {
        "en": 23.0,
        "de": 10.7,
        "es": 13.2,
        "fr": 12.2,
        "it": 8.8,
        "ja": 9.1,
        "ko": 5.0,
        "zh": 7.5,
    },
    "samsum": {
        "en": 38.2,
        "de": 28.6,
        "es": 31.3,
        "fr": 30.2,
        "it": 28.3,
        "ja": 27.5,
        "ko": 14.0,
        "zh": 22.7,
    },
    "content_rephrasing": {
        "en": 58.1,
        "de": 43.8,
        "es": 44.4,
        "fr": 46.2,
        "it": 42.8,
        "ja": 39.4,
        "ko": 30.5,
        "zh": 35.2,
    },
}


single_lora_scores_qwen = {
    "text-correction": {
        "en": 38.6,
        "de": 23.0,
        "es": 29.7,
        "fr": 36.8,
        "it": 31.6,
        "ja": 26.4,
        "ko": 21.5,
        "zh": 63.2,
    },
    "squad": {
        "en": 68.0,
        "de": 42.4,
        "es": 48.7,
        "fr": 38.7,
        "it": 42.0,
        "ja": 27.9,
        "ko": 23.3,
        "zh": 29.1,
    },
    "persona-chat-synthetic": {
        "en": 22.3,
        "de": 12.8,
        "es": 14.7,
        "fr": 14.3,
        "it": 11.6,
        "ja": 11.6,
        "ko": 6.2,
        "zh": 6.6,
    },
    "samsum": {
        "en": 38.6,
        "de": 28.9,
        "es": 32.6,
        "fr": 31.3,
        "it": 30.2,
        "ja": 26.7,
        "ko": 15.2,
        "zh": 26.1,
    },
    "content_rephrasing": {
        "en": 43.5,
        "de": 33.9,
        "es": 38.5,
        "fr": 30.3,
        "it": 32.2,
        "ja": 32.8,
        "ko": 16.8,
        "zh": 27.4,
    },
}


single_lora_scores = {
    "llama": single_lora_scores_llama,
    "qwen": single_lora_scores_qwen,
}


def get_filenames(directory):
    """List all subdirectories of `directory` (one per run/output)."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    )


def compute_s_score(filename, output_dir, lora_scores):
    with open(os.path.join(output_dir, filename, "all_results.json")) as f:
        d = json.load(f)

    step = 40
    s_score = 0
    for metric in metrics_order_to_print:
        for lang in ["en", "de", "es", "fr", "it", "ko", "ja", "zh"]:

            single_lora_score_task_lang = lora_scores[metric][lang]

            if (metric == "text-correction") and (lang != "en"):
                m = "text-correction-org-lang"
            else:
                m = metric

            key = f'test_{m + "_" + lang}_{step}_{test_metrics[metric]}'
            perf_task_lang = d[key]

            if metric != "squad":
                perf_task_lang *= 100

            s_score += perf_task_lang / single_lora_score_task_lang

    return s_score / 40


def main():
    parser = argparse.ArgumentParser(
        description="Compute the S-score for every output folder."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["llama", "qwen"],
        help="Model to use for selecting the reference single-LoRA scores.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help=(
            "Directory containing one subfolder per run, each with an "
            "all_results.json file (default: output). Subfolder names are "
            "automatically used as filenames."
        ),
    )
    args = parser.parse_args()

    lora_scores = single_lora_scores[args.model]
    filenames = get_filenames(args.output_dir)

    for filename in filenames:
        s_score = compute_s_score(filename, args.output_dir, lora_scores)
        print(f"{filename} , {s_score}")


if __name__ == "__main__":
    main()
