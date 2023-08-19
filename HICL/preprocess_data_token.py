import argparse
import os
import datasets
import json
from transformers import  AutoTokenizer
import copy
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='./', type=str, required=False)
parser.add_argument("--dataset_path", default='./', type=str, required=False)
parser.add_argument("--task_name", default='hash_database', type=str, required=False)
parser.add_argument("--use_slow_tokenizer", action="store_true")
parser.add_argument("--tokenizer_name", default='albertan017/hashencoder', type=str, required=False)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--preprocessing_num_workers", default=10, type=int)
parser.add_argument("--overwrite_cache", type=bool, default=False)
parser.add_argument("--write_name", default='', type=str, required=False)


def tokenization(args):
    data_files = {}
    data_files["train"] = args.dataset_path + args.task_name + '.txt'
    raw_datasets = datasets.load_dataset('text', data_files=data_files)
    # raw_datasets["train"] = raw_datasets["train"].shuffle()
    # Load pretrained tokenizer
    if 'bertweet' in args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer,
                                                  normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    padding = False

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=False,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        # remove_columns=[text_column_name],
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset line_by_line",
    )

    # tokenized_datasets.save_to_disk(args.output_dir)
    return tokenized_datasets

if __name__ == "__main__":
    args = parser.parse_args()
    args_tmp = copy.deepcopy(args)
    tokenized_datasets = tokenization(args_tmp)
    tokenized_datasets.save_to_disk(args_tmp.output_dir + args.task_name + args.write_name)
