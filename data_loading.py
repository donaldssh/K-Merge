import json
import os
import random

import markdown
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

RANDOM_SEED = 42
ALL_TONES = ["professional", "casual", "witty", "paraphrase"]


def parse_dataset_spec(dataset_spec):
    # the syntax for multi-dataset settings is e.g. a+b+c,a+b+c,d
    datasets_per_split = dataset_spec.split(",")
    if len(datasets_per_split) == 1:
        dataset_names = datasets_per_split[0].split("+")
        dataset_dicts = {"train": [], "validation": [], "test": []}
        for split in ["train", "validation", "test"]:
            dataset_dicts[split] = dataset_names
    elif len(datasets_per_split) == 3:
        dataset_dicts = {"train": [], "validation": [], "test": []}
        dataset_names = set([])
        for datasets_split, split in zip(
            datasets_per_split, ["train", "validation", "test"]
        ):
            separated_datasets = datasets_split.split("+")
            for dataset in separated_datasets:
                dataset_names.add(dataset)
                dataset_dicts[split].append(dataset)
        dataset_names = list(dataset_names)
    else:
        raise ValueError("Dataset name is not in a suitable format: ", dataset_spec)

    return dataset_names, dataset_dicts


def load_persona_chat_synthetic(**kwargs):
    # google/Synthetic-Persona-Chat
    # we do not use the additional synthetic part
    dataset = load_dataset("google/Synthetic-Persona-Chat", trust_remote_code=True)
    inputs = {"train": [], "validation": [], "test": []}
    outputs = {"train": [], "validation": [], "test": []}

    for split in inputs.keys():
        for dialog in dataset[split]["Best Generated Conversation"]:
            # this is a list of sentences, split it into consecutive pairs
            if dialog:
                dialog_items = dialog.split("\n")
                if len(dialog_items) > 1:
                    for idx in range(len(dialog_items) - 1):
                        if len(dialog_items[idx]) > 0:
                            if (
                                dialog_items[idx][:8] == "User 1: "
                                and dialog_items[idx + 1][:8] == "User 2: "
                            ) or (
                                dialog_items[idx][:8] == "User 2: "
                                and dialog_items[idx + 1][:8] == "User 1: "
                            ):
                                inputs[split].append(dialog_items[idx][8:])
                                outputs[split].append(dialog_items[idx + 1][8:])

    dataset_dict = {}
    for split in inputs.keys():
        dataset = Dataset.from_dict({"input": inputs[split], "output": outputs[split]})
        dataset_dict[split] = dataset
    dataset_dict = DatasetDict(dataset_dict)
    return dataset_dict


def load_samsum(**kwargs):
    dataset = load_dataset("Samsung/samsum", trust_remote_code=True)
    inputs = {
        "train": dataset["train"]["dialogue"],
        "validation": dataset["validation"]["dialogue"],
        "test": dataset["test"]["dialogue"],
    }
    outputs = {
        "train": dataset["train"]["summary"],
        "validation": dataset["validation"]["summary"],
        "test": dataset["test"]["summary"],
    }
    dataset_dict = {}
    for split in inputs.keys():
        dataset = Dataset.from_dict({"input": inputs[split], "output": outputs[split]})
        dataset_dict[split] = dataset
    dataset_dict = DatasetDict(dataset_dict)
    return dataset_dict


def load_squad(**kwargs):
    def get_inputs(d):
        return [
            " ".join(["question:", q.lstrip(), "context:", c.lstrip()])
            for q, c in zip(d["question"], d["context"])
        ]

    def get_outputs(answers):
        return [a["text"][0] if len(a["text"]) > 0 else "" for a in answers]

    dataset = load_dataset("squad", trust_remote_code=True)
    # dataset has only training and validation (latter is used for testing).
    # we split the training into train and validation
    dataset_split = dataset["train"].train_test_split()

    inputs = {
        "train": get_inputs(dataset_split["train"]),
        "validation": get_inputs(dataset_split["test"]),
        "test": get_inputs(dataset["validation"]),
    }
    outputs = {
        "train": get_outputs(dataset_split["train"]["answers"]),
        "validation": get_outputs(dataset_split["test"]["answers"]),
        "test": get_outputs(dataset["validation"]["answers"]),
    }
    dataset_dict = {}
    for split in inputs.keys():
        dataset = Dataset.from_dict(
            {
                "input": inputs[split],
                "output": outputs[split],
            }
        )
        dataset_dict[split] = dataset
    dataset_dict = DatasetDict(dataset_dict)
    return dataset_dict


def load_text_correction_dataset(**kwargs):
    """
    Write & Improve text correction dataset.
    """
    local_repo = kwargs.get("local_repo", "./data")
    with open(f"{local_repo}/wi_text_correction.json", "r") as f:
        data = json.load(f)

    ds_train = Dataset.from_dict(data["train"])
    ds_val = Dataset.from_dict(data["validation"])
    ds_test = Dataset.from_dict(data["test"])

    return DatasetDict(train=ds_train, validation=ds_val, test=ds_test)


def load_text_correction_dataset_org_lang(**kwargs):
    """
    Text correction dataset collected in the original language.
    """

    def read_md(filename):
        data = open(filename, "r")
        data = markdown.markdown(data.read()).split("\n")
        data = str_2_dict(data)
        return data

    def str_2_dict(data):
        res = {}
        for i in range(len(data) // 2):
            sample_id = data[2 * i].replace("</h3>", "").replace("<h3>", "")
            sample_text = data[2 * i + 1].replace("</p>", "").replace("<p>", "")
            res[sample_id] = sample_text
        return res

    def combine_src_tgt(org, ref):
        data = {"src": [], "tgt": []}
        for text_id in org:
            data["src"].append(org[text_id])
            data["tgt"].append(ref[text_id])
        return data

    dataset_lang = kwargs.get("dataset_language", "en")

    def split_data(data):
        data_split = {"src": [], "tgt": []}
        for sample in data:
            for edit in sample["edits"]:
                data_split["src"].append(edit["src"]["text"])
                data_split["tgt"].append(edit["tgt"]["text"])
        return data_split

    if dataset_lang == "it":
        # 651 training samples
        org_train = read_md("data/gec-it/it-merlin-orig-train.md")
        ref_train = read_md("data/gec-it/it-merlin-ref1-train.md")
        train = combine_src_tgt(org_train, ref_train)
        train_ds = Dataset.from_dict({"src": train["src"], "tgt": train["tgt"]})
        # 572 training and 79 validation samples
        train_ds = train_ds.train_test_split(
            test_size=0.12, shuffle=False, seed=RANDOM_SEED
        )

        # 81 test samples
        org_test = read_md("data/gec-it/it-merlin-orig-dev.md")
        ref_test = read_md("data/gec-it/it-merlin-ref1-dev.md")
        test = combine_src_tgt(org_test, ref_test)

        # split into sentences and remove if correct

        inputs = {
            "train": train_ds["train"]["src"],
            "validation": train_ds["test"]["src"],
            "test": test["src"],
        }
        outputs = {
            "train": train_ds["train"]["tgt"],
            "validation": train_ds["test"]["tgt"],
            "test": test["tgt"],
        }
    elif dataset_lang == "zh":

        def split_lines(lines):
            inputs = []
            outputs = []
            for line in lines:
                # if len(line) < 200:
                sentences = line.split("\t")
                inputs.append(sentences[1])
                outputs.append(sentences[2])
            return {"src": inputs, "tgt": outputs}

        # load the data
        domains = ["law", "med", "odw"]
        train_lines = []
        for domain in domains:
            train_lines += (
                open(f"ECSpell/Data/domains_data/{domain}.train", "r", encoding="utf-8")
                .read()
                .strip()
                .split("\n")
            )
        test_lines = []
        for domain in domains:
            test_lines += (
                open(f"ECSpell/Data/domains_data/{domain}.test", "r", encoding="utf-8")
                .read()
                .strip()
                .split("\n")
            )

        N = len(test_lines)
        indices = list(range(N))
        random.Random(RANDOM_SEED).shuffle(indices)

        # split into training, validation and test
        validation_lines = [test_lines[i] for i in indices[: int(len(indices) * 0.5)]]
        test_lines = [test_lines[i] for i in indices[int(len(indices) * 0.5) :]]

        # split into error and corrections
        train = split_lines(train_lines)
        validation = split_lines(validation_lines)
        test = split_lines(test_lines)

        inputs = {
            "train": train["src"],
            "validation": validation["src"],
            "test": test["src"],
        }
        outputs = {
            "train": train["tgt"],
            "validation": validation["tgt"],
            "test": test["tgt"],
        }
    else:
        with open("data/github-typo-corpus.v1.0.0.jsonl") as f:
            data = [json.loads(line) for line in f]

        github_dataset_lang_mapping = {
            "de": "deu",
            "es": "spa",
            "fr": "fra",
            "ja": "jpn",
            "ko": "kor",
        }
        # filter the data
        data = [
            sample
            for sample in data
            if sample["edits"][0]["src"]["lang"]
            == github_dataset_lang_mapping[dataset_lang]
        ]

        N = len(data)
        indices = list(range(N))
        random.Random(RANDOM_SEED).shuffle(indices)

        # split into training, validation and test
        train_data = [data[i] for i in indices[: int(len(indices) * 0.6)]]
        validation_data = [data[i] for i in indices[int(len(indices) * 0.6) :]]
        test_data = validation_data[: len(validation_data) // 2]
        validation_data = validation_data[len(validation_data) // 2 :]

        train = split_data(train_data)
        validation = split_data(validation_data)
        test = split_data(test_data)

        inputs = {
            "train": train["src"],
            "validation": validation["src"],
            "test": test["src"],
        }
        outputs = {
            "train": train["tgt"],
            "validation": validation["tgt"],
            "test": test["tgt"],
        }

    dataset_dict = {}
    for split in inputs.keys():
        dataset = Dataset.from_dict({"input": inputs[split], "output": outputs[split]})
        dataset_dict[split] = dataset
    dataset_dict = DatasetDict(dataset_dict)
    return dataset_dict


def load_content_rephrasing(**kwargs):
    tone = kwargs.get("tone", "original")
    local_repo = kwargs.get("local_repo", "./data")
    if tone == "original":
        dataset = load_dataset("facebook/content_rephrasing", trust_remote_code=True)
        inputs = {
            "train": [
                e[0].capitalize() + e[1:] for e in dataset["train"]["Rephrased Content"]
            ],
            "validation": [
                e[0].capitalize() + e[1:]
                for e in dataset["validation"]["Rephrased Content"]
            ],
            "test": [
                e[0].capitalize() + e[1:] for e in dataset["test"]["Rephrased Content"]
            ],
        }
        outputs = {
            "train": [
                e[0].capitalize() + e[1:] for e in dataset["train"]["Rephrased Content"]
            ],
            "validation": [
                e[0].capitalize() + e[1:]
                for e in dataset["validation"]["Rephrased Content"]
            ],
            "test": [
                e[0].capitalize() + e[1:] for e in dataset["test"]["Rephrased Content"]
            ],
        }
        dataset_dict = {}
        for split in inputs.keys():
            dataset = Dataset.from_dict(
                {"input": inputs[split], "output": outputs[split]}
            )
            dataset_dict[split] = dataset
        dataset_dict = DatasetDict(dataset_dict)
    else:
        dataset_dict = load_from_disk(
            os.path.join(local_repo, "content_rephrasing_" + tone)
        )

    return dataset_dict


DATASET_NAME_TO_LOADER = {
    "persona-chat-synthetic": load_persona_chat_synthetic,
    "samsum": load_samsum,
    "text-correction": load_text_correction_dataset,
    "text-correction-org-lang": load_text_correction_dataset_org_lang,
    "content_rephrasing": load_content_rephrasing,
    "squad": load_squad,
}


def load_own_data(dataset_spec="text-correction", **loader_kwargs):
    dataset_names, dataset_dicts = parse_dataset_spec(dataset_spec)

    loaded_dataset_dicts = {}
    for dataset_name in dataset_names:
        dataset_language = loader_kwargs.get("dataset_language", "en")
        local_repo = loader_kwargs.get("local_repo", "./data")
        if dataset_language != "en" and dataset_name != "text-correction-org-lang":
            if dataset_name == "content_rephrasing":
                tone = loader_kwargs.get("tone", "professional")
                if tone == "generic":
                    dd = DatasetDict()
                    for tone in ALL_TONES:
                        dataset_tone_lang = load_from_disk(
                            f"{local_repo}/{dataset_name}_{tone}_m100_{dataset_language}"
                        )
                        if tone == ALL_TONES[0]:
                            for key in dataset_tone_lang:
                                dd[key] = dataset_tone_lang[key]
                            del dataset_tone_lang
                        else:
                            for key in dataset_tone_lang:
                                dd[key] = concatenate_datasets(
                                    [ddd[key] for ddd in [dd, dataset_tone_lang]]
                                )
                    loaded_dataset_dicts[dataset_name] = dd
                else:
                    loaded_dataset_dicts[dataset_name] = load_from_disk(
                        f"{local_repo}/{dataset_name}_{tone}_m100_{dataset_language}"
                    )
            else:
                loaded_dataset_dicts[dataset_name] = load_from_disk(
                    f"{local_repo}/{dataset_name}_m100_{dataset_language}"
                )
        else:
            if dataset_name == "content_rephrasing":
                tone = loader_kwargs.get("tone", "professional")
                if tone == "generic":
                    dd = DatasetDict()
                    for tone in ALL_TONES:
                        loader_kwargs["tone"] = tone
                        dataset_tone = DATASET_NAME_TO_LOADER[dataset_name](
                            **loader_kwargs
                        )
                        if tone == ALL_TONES[0]:
                            for key in dataset_tone:
                                dd[key] = dataset_tone[key]
                            del dataset_tone
                        else:
                            for key in dataset_tone:
                                dd[key] = concatenate_datasets(
                                    [ddd[key] for ddd in [dd, dataset_tone]]
                                )
                    loaded_dataset_dicts[dataset_name] = dd
                else:
                    loaded_dataset_dicts[dataset_name] = DATASET_NAME_TO_LOADER[
                        dataset_name
                    ](**loader_kwargs)
            else:
                loaded_dataset_dicts[dataset_name] = DATASET_NAME_TO_LOADER[
                    dataset_name
                ](**loader_kwargs)

    return DatasetDict(
        {
            "train": concatenate_datasets(
                [
                    loaded_dataset_dicts[dataset_name]["train"]
                    for dataset_name in dataset_dicts["train"]
                ]
            ),
            "validation": concatenate_datasets(
                [
                    loaded_dataset_dicts[dataset_name]["validation"]
                    for dataset_name in dataset_dicts["validation"]
                ]
            ),
            "test": concatenate_datasets(
                [
                    loaded_dataset_dicts[dataset_name]["test"]
                    for dataset_name in dataset_dicts["test"]
                ]
            ),
        }
    )
