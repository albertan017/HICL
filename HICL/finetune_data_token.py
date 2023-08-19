import argparse
import os
import datasets
import json
from transformers import  AutoTokenizer
import copy
from accelerate import Accelerator
#'eval-emoji,eval-emotion,eval-hate,eval-irony,eval-offensive,eval-sentiment,eval-stance/abortion,eval-stance/atheism,eval-stance/climate,eval-stance/feminist,eval-stance/hillary'
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='../finetune/data/', type=str, required=False, help="The output directory where the data will be written.")
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset path")
parser.add_argument("--task_name", default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument('--method',default='token',type=str)
parser.add_argument("--tokenizer_name", default='albertan017/hashencoder', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")

def preprocess(text):
    preprocessed_text = []
    for t in text.split():
        if len(t) > 1:
            t = '@user' if t[0] == '@' and t.count('@') == 1 else t
            t = 'http' if t.startswith('http') else t
        preprocessed_text.append(t)
    return ' '.join(preprocessed_text)

def tokenization(args):
    data_files = {}
    data_files["train"] = args.dataset_path + args.task_name + '/train.json'
    data_files["dev"] = args.dataset_path + args.task_name + '/dev.json'
    data_files["test"] = args.dataset_path + args.task_name + '/test.json'
    raw_datasets = datasets.load_dataset('json', data_files=data_files)
    raw_datasets["train"] = raw_datasets["train"].shuffle()
    raw_datasets["dev"] = raw_datasets["dev"].shuffle()
    raw_datasets["test"] = raw_datasets["test"].shuffle()
    # Load pretrained tokenizer
    if 'bertweet' in args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # First we tokenize all the texts.

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    padding = False

    def tokenize_function(examples):
        if 'cardiffnlp' in args.tokenizer_name:
            examples[text_column_name] = [
                preprocess(line) for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
        else:
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=False,
            return_token_type_ids=True
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset line_by_line",
    )

    # tokenized_datasets.save_to_disk(args.output_dir)
    return tokenized_datasets

if __name__ == "__main__":
    args = parser.parse_args()

    for task in args.task_name.split(','):
        args_tmp = copy.deepcopy(args)
        args_tmp.task_name = task
        tokenized_datasets = tokenization(args_tmp)
        tokenized_datasets.save_to_disk(args_tmp.dataset_path + args_tmp.task_name + '/' + args_tmp.method)