import re
import string
from collections import Counter

import errant
import evaluate
import fugashi
import jieba
import numpy as np
import torch
from ChERRANT.modules.annotator import Annotator
from ChERRANT.modules.tokenizer import Tokenizer as Tokenizer_ChERRANT
from peft import PeftModel
from rouge_score import rouge_scorer
from rouge_score.tokenize import SPACES_RE
from rouge_score.tokenizers import Tokenizer
from sumeval.metrics.rouge import RougeCalculator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from prompts import PROMPTS

ja_tagger = fugashi.Tagger()


class NonAlphaNumericSupportTokenizer(Tokenizer):
    """
    >>> NonAlphaNumericSupportTokenizer().tokenize("いぬ ねこ")
    ['いぬ', 'ねこ']
    """

    def tokenize(self, text):
        return SPACES_RE.split(text.lower())


rouge_scorer_obt = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rouge3", "rougeL"], use_stemmer=True
)
rouge_scorer_obt_ko = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rouge3", "rougeL"],
    tokenizer=NonAlphaNumericSupportTokenizer(),
)
rouge_calculator_ja = RougeCalculator(stopwords=True, lang="ja")
rouge_calculator_zh = RougeCalculator(stopwords=True, lang="zh")
qa_squad = evaluate.load("squad")
EOC_FORMAT = "\n\n### END"


def rouge_score_ja_zh_score(ref, pred, lang):
    if lang == "ja":
        ja_zh_scorer = rouge_calculator_ja
    else:
        ja_zh_scorer = rouge_calculator_zh
    scores = {
        "rouge1": ja_zh_scorer.rouge_n(summary=pred, references=ref, n=1),
        "rouge2": ja_zh_scorer.rouge_n(summary=pred, references=ref, n=2),
        "rouge3": ja_zh_scorer.rouge_n(summary=pred, references=ref, n=3),
        "rougeL": ja_zh_scorer.rouge_l(summary=pred, references=ref),
    }
    return scores


def print_trainable_parameters(model):
    # print the number of trainable parameters in the model

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || " f"all params: {all_param} || ")


def get_task_prompt(dataset_name, dataset_language, tone):
    # get prompt for the given task

    if dataset_name in ["tone", "content_rephrasing"]:
        return PROMPTS[dataset_name](dataset_language, tone)
    elif dataset_name == "text-correction-org-lang":
        return PROMPTS["text-correction"](dataset_language)
    else:
        return PROMPTS[dataset_name](dataset_language)


def subsample_dataset(dataset, split, max_samples):
    # function to subsample the dataset to a maximum number of samples

    dataset = dataset[split]
    if max_samples is not None and len(dataset) > max_samples:
        idcs = (
            np.random.RandomState(seed=42)
            .permutation(len(dataset))[:max_samples]
            .tolist()
        )
        dataset = dataset.select(idcs)
    return dataset


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_tone_adj_pipeline(args):
    # get pipeline to perform tone adjustment

    model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
    peft_model_id = "llm-toys/RedPajama-INCITE-Base-3B-v1-paraphrase-tone"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side=args.padding_side, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=get_bnb_config(),
    )

    model = PeftModel.from_pretrained(model, peft_model_id)

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        eos_token_id=tokenizer(" END")["input_ids"],
        batch_size=args.per_device_eval_batch_size,
        padding=True,
        truncation=True,
        return_full_text=False,
        do_sample=False,
        num_return_sequences=1,
    )


def _blended_rouge(rouge: dict, lang: str):
    # Mixes together r1, r2, r3 according to https://arxiv.org/pdf/2106.02017.pdf

    if lang in ["ja", "zh"]:
        r1 = rouge["rouge1"] / 6
        r2 = rouge["rouge2"] / 3
        r3 = rouge["rouge3"] / 2
    else:
        r1 = rouge["rouge1"].fmeasure / 6
        r2 = rouge["rouge2"].fmeasure / 3
        r3 = rouge["rouge3"].fmeasure / 2
    return r1 + r2 + r3


def smart_reply_rouge(targets, preds, lang):
    # Computes blended rouge for a batch of samples

    def reply_rouge_scorer_score(t, p):
        if lang in ["ja", "zh"]:
            return rouge_score_ja_zh_score(t, p, lang)
        elif lang == "ko":
            return rouge_scorer_obt_ko.score(t, p)
        else:
            return rouge_scorer_obt.score(t, p)

    scores = []
    for t, P in zip(targets, preds):
        if type(P) == list:
            score = max(
                [_blended_rouge(reply_rouge_scorer_score(t, p), lang) for p in P]
            )
        else:
            score = _blended_rouge(reply_rouge_scorer_score(t, P), lang)
        scores.append(score)

    return scores


def compute_qa_squad(preds, refs, lang):
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(prediction, ground_truth):
        if lang == "ja":
            prediction_tokens = [word.surface for word in ja_tagger(prediction)]
            ground_truth_tokens = [word.surface for word in ja_tagger(ground_truth)]
        elif lang == "zh":
            prediction_tokens = [word[0] for word in jieba.tokenize(prediction)]
            ground_truth_tokens = [word[0] for word in jieba.tokenize(ground_truth)]
        else:
            prediction_tokens = normalize_answer(prediction).split()
            ground_truth_tokens = normalize_answer(ground_truth).split()

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(prediction, ground_truth):
        return normalize_answer(prediction) == normalize_answer(ground_truth)

    f1 = exact_match = total = 0
    for pred, ref in zip(preds, refs):
        exact_match += exact_match_score(pred, ref)
        f1 += f1_score(pred, ref)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


def compute_f05_errant_score(inputs, preds, refs, lang):
    def simplify_edits(sent, lang):
        out_edits = []

        if lang == "zh":
            # Get the edit lines from an m2 block.
            edits = sent.split("\n")
            # Loop through the edits
            for edit in edits:
                # Preprocessing
                if edit.startswith("A "):
                    edit = edit[2:].split("|||")  # Ignore "A " then split.
                    span = edit[0].split()
                    start = int(span[0])
                    end = int(span[1])
                    cat = edit[1]
                    cor = edit[2].replace(" ", "")
                    coder = int(edit[-1])
                    out_edit = [start, end, cat, cor, coder]
                    out_edits.append(out_edit)
        else:
            # Get the edit lines from an m2 block.
            edits = sent.split("\n")[1:]
            # Loop through the edits
            for edit in edits:
                # Preprocessing
                edit = edit[2:].split("|||")  # Ignore "A " then split.
                span = edit[0].split()
                start = int(span[0])
                end = int(span[1])
                cat = edit[1]
                cor = edit[2]
                coder = int(edit[-1])
                out_edit = [start, end, cat, cor, coder]
                out_edits.append(out_edit)
        return out_edits

    def process_edits(edits, lang):
        coder_dict = {}
        if lang == "zh":
            dt = False
            ds = False
            # Add an explicit noop edit if there are no edits.
            if not edits:
                edits = [[-1, -1, "noop", "-NONE-", 0]]
            # Loop through the edits
            for edit in edits:
                # Name the edit elements for clarity
                start = edit[0]
                end = edit[1]
                cat = edit[2]
                cor = edit[3]
                coder = edit[4]
                # Add the coder to the coder_dict if necessary
                if coder not in coder_dict:
                    coder_dict[coder] = {}

                # Optionally apply filters based on args
                # 1. UNK type edits are only useful for detection, not correction.
                if not dt and not ds and cat == "UNK":
                    continue

                if (start, end, cor) in coder_dict[coder].keys():
                    coder_dict[coder][(start, end, cor)].append(cat)
                else:
                    coder_dict[coder][(start, end, cor)] = [cat]
        else:
            # Add an explicit noop edit if there are no edits.
            if not edits:
                edits = [[-1, -1, "noop", "-NONE-", 0]]
            # Loop through the edits
            for edit in edits:
                # Name the edit elements for clarity
                start = edit[0]
                end = edit[1]
                cat = edit[2]
                cor = edit[3]
                coder = edit[4]
                # Add the coder to the coder_dict if necessary
                if coder not in coder_dict:
                    coder_dict[coder] = {}

                if (start, end, cor) in coder_dict[coder].keys():
                    coder_dict[coder][(start, end, cor)].append(cat)
                else:
                    coder_dict[coder][(start, end, cor)] = [cat]
        return coder_dict

    def evaluate_edits(hyp_dict, ref_dict, best, lang):
        best_tp, best_fp, best_fn, best_f = 0, 0, 0, -1
        beta = 0.5
        if lang == "zh":
            # skip not annotatable sentence
            if len(ref_dict.keys()) == 1:
                ref_id = list(ref_dict.keys())[0]
                if len(ref_dict[ref_id].keys()) == 1:
                    cat = list(ref_dict[ref_id].values())[0][0]
                    if cat == "NA":
                        best_dict = {"tp": best_tp, "fp": best_fp, "fn": best_fn}
                        return best_dict

            # Compare each hyp and ref combination
            for hyp_id in hyp_dict.keys():
                for ref_id in ref_dict.keys():
                    # Get the local counts for the current combination.
                    tp, fp, fn = compareEdits(hyp_dict[hyp_id], ref_dict[ref_id])
                    # Compute the global sentence scores
                    f = computeFScore(
                        tp + best["tp"], fp + best["fp"], fn + best["fn"], beta
                    )
                    if (
                        (f > best_f)
                        or (f == best_f and tp > best_tp)
                        or (f == best_f and tp == best_tp and fp < best_fp)
                        or (
                            f == best_f
                            and tp == best_tp
                            and fp == best_fp
                            and fn < best_fn
                        )
                    ):
                        best_tp, best_fp, best_fn = tp, fp, fn
                        best_f = f
        else:
            for hyp_id in hyp_dict.keys():
                for ref_id in ref_dict.keys():
                    # Get the local counts for the current combination.
                    tp, fp, fn = compareEdits(hyp_dict[hyp_id], ref_dict[ref_id])
                    # Compute the global sentence scores
                    f = computeFScore(
                        tp + best["tp"], fp + best["fp"], fn + best["fn"], beta
                    )
                    if (
                        (f > best_f)
                        or (f == best_f and tp > best_tp)
                        or (f == best_f and tp == best_tp and fp < best_fp)
                        or (
                            f == best_f
                            and tp == best_tp
                            and fp == best_fp
                            and fn < best_fn
                        )
                    ):
                        best_tp, best_fp, best_fn = tp, fp, fn
                        best_f = f
        # Save the best TP, FP and FNs as a dict, and return this
        best_dict = {"tp": best_tp, "fp": best_fp, "fn": best_fn}
        return best_dict

    def compareEdits(hyp_edits, ref_edits):
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives

        for h_edit, h_cats in hyp_edits.items():
            # noop hyp edits cannot be TP or FP
            if h_cats[0] == "noop":
                continue
            # TRUE POSITIVES
            if h_edit in ref_edits.keys():
                tp += len(ref_edits[h_edit])  # Use ref dict for TP
            # FALSE POSITIVES
            else:
                fp += len(h_cats)
        for r_edit, r_cats in ref_edits.items():
            # noop ref edits cannot be FN
            if r_cats[0] == "noop":
                continue
            # FALSE NEGATIVES
            if r_edit not in hyp_edits.keys():
                fn += len(r_cats)
        return tp, fp, fn

    def computeFScore(tp, fp, fn, beta):
        p = float(tp) / (tp + fp) if fp else 1.0
        r = float(tp) / (tp + fn) if fn else 1.0
        f = float((1 + (beta**2)) * p * r) / (((beta**2) * p) + r) if p + r else 0.0
        return f

    def noop_edit(id=0):
        return "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||" + str(id)

    def convert_to_m2_format(original, corrected, lang):
        if lang == "zh":

            def annotate(source, target):
                source = "".join(source.strip().split())
                target = "".join(target.strip().split())
                source_tokenized, target_tokenized = tokenizer(source), tokenizer(
                    target
                )

                output_str = ""
                out, cors = annotator(source_tokenized, target_tokenized, 0)
                output_str += "".join(out[:-1])
                return output_str

            granularity = "char"
            annotator = Annotator.create_default(granularity, "all")
            tokenizer = Tokenizer_ChERRANT(granularity, "cuda", False, False)

            count = 0
            batch_size = 4
            sentence_set = set()
            sentence_to_tokenized = {}
            for orig, cor in zip(original, corrected):
                # Loop through the corrected texts. note: cor is a list for output and a string for ref
                if type(cor) == list:
                    cor = cor[0].strip()
                else:
                    cor = cor.strip()
                sentence_set.add(cor)

            batch = []
            for sent in sentence_set:
                count += 1
                if sent:
                    batch.append(sent)
                if count % batch_size == 0:
                    results = tokenizer(batch)
                    for s, r in zip(batch, results):
                        sentence_to_tokenized[s] = r  # Get tokenization map.
                    batch = []
            if batch:
                results = tokenizer(batch)
                for s, r in zip(batch, results):
                    sentence_to_tokenized[s] = r  # Get tokenization map.

            out_m2_list = []
            for orig, cor in zip(original, corrected):
                # Loop through the corrected texts. note: cor is a list for output and a string for ref
                if type(cor) == list:
                    cor = cor[0].strip()
                else:
                    cor = cor.strip()

                ret = annotate(orig, cor)
                out_m2_list.append(ret)

        else:
            merge = "rules"
            lev = False
            tok = False
            annotator = errant.load("en")
            out_m2_list = []
            for orig, cor in zip(original, corrected):
                out_m2 = ""
                orig = annotator.parse(orig, False)
                # Write orig to the output m2 file
                out_m2 += " ".join(["S"] + [token.text for token in orig]) + "\n"

                # Loop through the corrected texts. note: cor is a list for output and a string for ref
                if type(cor) == list:
                    cor = cor[0].strip()
                else:
                    cor = cor.strip()

                # If the texts are the same, write a noop edit
                if orig.text.strip() == cor:
                    out_m2 += noop_edit(0) + "\n"
                # Otherwise, do extra processing
                else:
                    # Parse cor with spacy
                    cor = annotator.parse(cor, tok)
                    # Align the texts and extract and classify the edits
                    edits = annotator.annotate(orig, cor, lev, merge)
                    # Loop through the edits
                    for edit in edits:
                        # Write the edit to the output m2 file
                        out_m2 += edit.to_m2(0) + "\n"
                out_m2_list.append(out_m2.strip())
        return out_m2_list

    text_correct_prompt = PROMPTS["text-correction"](lang)
    filtered_inputs = []
    for input in inputs:
        if text_correct_prompt in input:
            input = input.split(text_correct_prompt)[1]
        filtered_inputs.append(input)

    preds_m2 = convert_to_m2_format(filtered_inputs, preds, lang)
    refs_m2 = convert_to_m2_format(filtered_inputs, refs, lang)

    # Store global corpus level best counts here
    best_dict = Counter({"tp": 0, "fp": 0, "fn": 0})
    # Process each sentence
    sents = zip(preds_m2, refs_m2)

    for sent in sents:
        # Simplify the edits into lists of lists
        hyp_edits = simplify_edits(sent[0], lang)
        ref_edits = simplify_edits(sent[1], lang)
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits, lang)
        ref_dict = process_edits(ref_edits, lang)
        # Evaluate edits and get best TP, FP, FN hyp+ref combo.
        count_dict = evaluate_edits(hyp_dict, ref_dict, best_dict, lang)
        # Merge these dicts with best_dict and best_cats
        best_dict += Counter(count_dict)
    return computeFScore(best_dict["tp"], best_dict["fp"], best_dict["fn"], 0.5)


def compute_task_metrics(references, predictions, inputs, dataset, dataset_language):
    # Main function to handle computation of task metrics
    # References are a list of sentences, while predictions are a list of lists of sentences.

    scores = {"n_examples": len(references)}

    if dataset in ["persona-chat-synthetic"]:
        scores["weighted_rouge"] = np.mean(
            smart_reply_rouge(references, predictions, dataset_language)
        ).item()
    elif dataset in [
        "samsum",
        "content_rephrasing",
    ]:
        if dataset_language == "ko":
            rouge_scores = [
                rouge_scorer_obt_ko.score(ref, pred[0])
                for ref, pred in zip(references, predictions)
            ]
        elif dataset_language in ["ja", "zh"]:
            rouge_scores = [
                rouge_score_ja_zh_score(ref, pred[0], dataset_language)
                for ref, pred in zip(references, predictions)
            ]
        else:
            rouge_scores = [
                rouge_scorer_obt.score(ref, pred[0])
                for ref, pred in zip(references, predictions)
            ]

        if dataset_language in ["ja", "zh"]:
            rouge = {
                metric: np.mean([score[metric] for score in rouge_scores]).item()
                for metric in ["rouge1", "rouge2", "rougeL"]
            }
        else:
            rouge = {
                metric: np.mean(
                    [score[metric].fmeasure for score in rouge_scores]
                ).item()
                for metric in ["rouge1", "rouge2", "rougeL"]
            }
        scores.update(rouge)

    elif dataset in ["text-correction", "text-correction-org-lang"]:
        scores["f05"] = compute_f05_errant_score(
            inputs, predictions, references, dataset_language
        )
    elif dataset == "squad":
        predictions = [e[0] for e in predictions]
        squad = compute_qa_squad(predictions, references, dataset_language)
        scores.update(squad)

    return scores
