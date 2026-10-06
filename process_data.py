import argparse
import copy
import json
import os

import nltk
from datasets import DatasetDict, load_dataset
from transformers import pipeline

from data_loading import load_own_data
from utils import EOC_FORMAT, get_tone_adj_pipeline

RANDOM_SEED = 42


def translate_dataset(args, dataset):
    # translate the full dataset, both inputs and outputs, into the selected language

    dataset_lang = copy.deepcopy(dataset)
    lang = args.target_language
    dataset_name = args.dataset

    if lang in ["de", "es", "fr", "it"]:
        model = f"Helsinki-NLP/opus-mt-en-{lang}"
    else:
        model = "facebook/m2m100_418M"

    translator = pipeline(
        task="translation",
        model=model,
        device="cuda",
        batch_size=4,
    )
    for split in ["train", "validation", "test"]:
        print("split =", split, flush=True)

        for part in ["input", "output"]:
            print("part =", part, flush=True)

            original_text = dataset[split][part]
            translated_text = translator(
                original_text,
                max_length=512,
                truncation=True,
                src_lang="en",
                tgt_lang=lang,
            )
            translated_text = [e["translation_text"] for e in translated_text]
            dataset_lang[split] = (
                dataset_lang[split]
                .remove_columns(part)
                .add_column(part, translated_text)
            )

    if dataset_name == "content_rephrasing" and args.tone and args.tone != "generic":
        save_name = f"{dataset_name}_{args.tone}_m100_{lang}"
    else:
        save_name = f"{dataset_name}_m100_{lang}"
    dataset_lang.save_to_disk(os.path.join(args.local_repo, save_name))


def change_tone_of_dataset(args, dataset):
    # change the tone of the dataset's outputs to the selected tone

    modifier = get_tone_adj_pipeline(args)
    for split in ["train", "validation", "test"]:
        original_outputs = [
            f"### Instruction:\nChange the tone of the following Input sentence to {args.tone}.\n\n### Input:\n{text}\n\n### Response:\n"
            for text in dataset[split]["output"]
        ]
        modified_outputs = modifier(original_outputs)
        targets = map(lambda x: x[0]["generated_text"], modified_outputs)
        targets = [
            (
                output[: output.find(EOC_FORMAT)].strip()
                if output.find(EOC_FORMAT) != -1
                else output
            )
            for output in targets
        ]
        dataset[split] = (
            dataset[split].remove_columns("output").add_column("output", targets)
        )
    dataset.save_to_disk(os.path.join(args.local_repo, args.dataset + "_" + args.tone))


def format_text(text):
    return (
        text.strip()
        .replace("\u00b4", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def process_text_correction_data():
    # process the text correction data

    dataset_url = "bea2019st/wi_locness"
    ds = load_dataset(dataset_url, "wi")
    ds_split = ds["train"].train_test_split(
        test_size=0.1, shuffle=False, seed=RANDOM_SEED
    )
    ds = DatasetDict(
        train=ds_split["train"], validation=ds_split["test"], test=ds["validation"]
    )
    processed_sentences_ds = {}
    for split in ["train", "validation", "test"]:
        input_texts = []
        output_texts = []
        for example in ds[split]:
            input_text = example["text"]
            output_text = example["text"]
            edits = example["edits"]
            net_shift = 0
            for i in range(len(edits["start"])):
                start = edits["start"][i] + net_shift
                end = edits["end"][i] + net_shift
                if edits["text"][i] is None:
                    output_text = output_text[:start] + output_text[end:]
                    net_shift -= end - start
                else:
                    output_text = (
                        output_text[:start] + edits["text"][i] + output_text[end:]
                    )
                    net_shift += len(edits["text"][i]) - (end - start)
            input_texts.append(input_text)
            output_texts.append(output_text.replace("  ", " "))

        filtered_input_sentences = []
        filtered_output_sentences = []
        for input_text, output_text in zip(input_texts, output_texts):
            input_sentences = nltk.sent_tokenize(input_text)
            output_sentences = nltk.sent_tokenize(output_text)
            input_sentences_set = set(nltk.sent_tokenize(input_text))

            for output_sentence in output_sentences:
                if output_sentence in input_sentences_set:
                    continue
                closest_input_sentence = ""
                max_overlap = 0
                for input_sentence in input_sentences:
                    # compute how many words overlap and take the one that has the largest overlap with the list of words in the output sentence
                    input_tokens = nltk.word_tokenize(input_sentence)
                    output_tokens = nltk.word_tokenize(output_sentence)
                    # compute the overlap taking into account repetitions
                    overlap = sum(
                        [
                            min(input_tokens.count(token), output_tokens.count(token))
                            for token in output_tokens
                        ]
                    )
                    if overlap > max_overlap:
                        max_overlap = overlap
                        closest_input_sentence = input_sentence

                closest_input_sentence = format_text(closest_input_sentence)
                output_sentence = format_text(output_sentence)
                if closest_input_sentence.isascii() and output_sentence.isascii():
                    filtered_input_sentences.append(closest_input_sentence)
                    filtered_output_sentences.append(output_sentence)

        processed_sentences_ds[split] = {
            "input": filtered_input_sentences,
            "output": filtered_output_sentences,
        }

    with open(os.path.join("data", "wi_text_correction.json"), "w") as f:
        json.dump(processed_sentences_ds, f)


def main():
    parser = argparse.ArgumentParser(description="Summarization Evaluation Script")
    parser.add_argument(
        "--dataset", type=str, default="samsum", help="Name of the dataset"
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default="",
        help="What should the translation target language be?",
    )
    parser.add_argument(
        "--tone", type=str, default="", help="Which tone to use for the tone task"
    )
    parser.add_argument(
        "--dataset_language",
        type=str,
        default="en",
        help="What is the language of the dataset",
    )
    parser.add_argument(
        "--local_repo",
        type=str,
        default="./data",
        help="Local directory where datasets are stored.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="For datasets that support multiple languages, the languages for which examples should be loaded",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="The evaluation batch size per GPU. Increase for better speed.",
    )
    parser.add_argument("--padding_side", type=str, default="left", help="Padding side")

    args = parser.parse_args()

    if args.dataset == "text-correction" and not args.target_language and not args.tone:
        process_text_correction_data()
        return

    if args.tone and not args.target_language:
        dataset = load_own_data(
            dataset_spec=args.dataset,
            local_repo=args.local_repo,
            languages=args.languages,
            dataset_language=args.dataset_language,
            tone="original",
        )
        change_tone_of_dataset(args, dataset)
        return

    dataset = load_own_data(
        dataset_spec=args.dataset,
        local_repo=args.local_repo,
        languages=args.languages,
        dataset_language=args.dataset_language,
        tone=args.tone,
    )

    if args.target_language:
        translate_dataset(args, dataset)


if __name__ == "__main__":
    main()
