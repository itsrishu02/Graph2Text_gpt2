import itertools
import json
import linecache
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import GPT2Tokenizer, PreTrainedTokenizer
from utils_graph2text import eval_bleu

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """Compute the label-smoothed negative log-likelihood loss."""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable) -> List:
    """List(map(f, x))"""
    return list(map(f, x))

def save_json(content, path, indent=4, **json_dump_kwargs):
    """Save a dictionary to a JSON file."""
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id."""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]

from sacrebleu import corpus_bleu

def calculate_bleu(predictions, references):
    return {"bleu": round(corpus_bleu(predictions, [references]).score, 4)}

from rouge_score import rouge_scorer, scoring

def calculate_rouge(predictions, references, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for pred, ref in zip(predictions, references):
        scores = scorer.score(pred, ref)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {
        "rouge1": round(result["rouge1"].mid.fmeasure * 100, 4),
        "rouge2": round(result["rouge2"].mid.fmeasure * 100, 4),
        "rougeL": round(result["rougeL"].mid.fmeasure * 100, 4),
    }

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    def __getitem__(self, index) -> Dict[str, str]:
        """Fetch source and target texts for a given index."""
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        target_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"Empty source line for index {index}"
        assert target_line, f"Empty target line for index {index}"
        return {"src_texts": source_line, "tgt_texts": target_line}

class Seq2SeqDataCollator:
    def __init__(self, tokenizer, max_source_length, max_target_length):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch):
        inputs = []
        targets = []

        # Parse graph input with XML-like tags
        for x in batch:
            source_line = x["src_texts"]
            target_line = x["tgt_texts"]

            if source_line.startswith('<H>'):
                # Split and clean the input
                head = source_line.split('<H> ')[-1].split(' <R>')[0].strip()
                relation = source_line.split('<R> ')[-1].split(' <T>')[0].strip()
                tail = source_line.split('<T> ')[-1].strip().rstrip('>')

                # Create a formatted input
                formatted_input = f"<H> {head} <R> {relation} <T> {tail}"
                formatted_target = target_line.strip()

                inputs.append(formatted_input)
                targets.append(formatted_target)

        # Combine input and target with a clear separator
        concatenated_texts = [
            f"{inp.strip()} -> {tgt.strip()} <|endoftext|>" for inp, tgt in zip(inputs, targets)
        ]

        # Tokenize concatenated sequences
        batch_encoding = self.tokenizer(
            concatenated_texts,
            max_length=self.max_source_length + self.max_target_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )


        labels = batch_encoding["input_ids"].clone()
        for i, inp in enumerate(inputs):
            formatted_input = f"{inp.strip()} ->"
            input_length = len(self.tokenizer(f"{formatted_input}", truncation=True)["input_ids"])
            labels[i, :input_length] = -100  # Mask input tokens


        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": batch_encoding["input_ids"],
            "attention_mask": batch_encoding["attention_mask"],
            "labels": labels,
        }


def freeze_params(model: torch.nn.Module):
    """Freeze parameters of the model."""
    for param in model.parameters():
        param.requires_grad = False

def write_txt_file(lines: List[str], path: str):
    """Write lines to a text file."""
    with open(path, "w") as file:
        for line in lines:
            file.write(line + "\n")
