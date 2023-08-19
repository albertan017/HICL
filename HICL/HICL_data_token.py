import argparse
import os
import datasets
import json
from transformers import  AutoTokenizer
import copy
from accelerate import Accelerator
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='../data/', type=str, required=False, help="The output directory where the data will be written.")
parser.add_argument("--dataset_path", default='../data/', type=str, required=False, help="dataset path")
parser.add_argument("--task_name", default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument('--method_hash',default='hicl_top100_sp',type=str,help="input suffix and save name")
parser.add_argument('--top',default='1',type=str,help="number of matched tweets to concatenate")
parser.add_argument("--tokenizer_name", default='albertan017/hashencoder', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")
parser.add_argument('--name',default='',type=str, help="save suffix")

def preprocess(text):
    preprocessed_text = []
    for t in text.split():
        if len(t) > 1:
            t = '@user' if t[0] == '@' and t.count('@') == 1 else t
            t = 'http' if t.startswith('http') else t
        preprocessed_text.append(t)
    return ' '.join(preprocessed_text)

def tokenization(args,tokenizer):

    data_files_hash = {}
    data_files_hash["train"] = args.dataset_path + args.task_name + '/train_'  + args.method_hash + '.json'
    data_files_hash["dev"] = args.dataset_path + args.task_name + '/dev_'  + args.method_hash + '.json'
    data_files_hash["test"] = args.dataset_path + args.task_name + '/test_'  + args.method_hash + '.json'
    raw_datasets = datasets.load_dataset('json', data_files=data_files_hash)

    remove_column_name = []
    for idx in range(100):
        if idx > args.top-1:
            remove_column_name.append('text'+str(idx))
    for sp in ['train', 'dev', 'test']:
        raw_datasets[sp] = raw_datasets[sp].remove_columns(remove_column_name)
        raw_datasets[sp] = raw_datasets[sp].shuffle()
    # First we tokenize all the texts.

    column_names = raw_datasets["train"].column_names
    print(column_names)

    padding = False

    def tokenize_function(examples):
        total = len(examples['text'])
        sentences = examples['text']
        for idx_tmp in range(args.top):
            sentences = sentences + examples['text'+str(idx_tmp)]
        sent_features = tokenizer(
            sentences,
            padding=padding,
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=False,
            return_token_type_ids=True
        )
        features = {}

        for key in sent_features:
            fea_one = []
            for idx_len in range(total):
                fea_one.append([])
                for idx_top in range(args.top+1):
                    fea_one[idx_len].append(sent_features[key][idx_top*total+idx_len])
            features[key] = fea_one


        features['labels'] = examples['labels']
        return features

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,#############need test
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset line_by_line",
    )

    # tokenized_datasets.save_to_disk(args.output_dir)
    return tokenized_datasets

if __name__ == "__main__":
    args = parser.parse_args()
    if 'bertweet' in args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    for task in args.task_name.split(','):
        for top in args.top.split(','):
            args_tmp = copy.deepcopy(args)
            args_tmp.task_name = task
            args_tmp.top = int(top)
            tokenized_datasets = tokenization(args_tmp,tokenizer)

            save_hash = args_tmp.method_hash.split('top')[0]
            tokenized_datasets.save_to_disk(args_tmp.dataset_path + args_tmp.task_name + '/hash_' \
                                            + save_hash + 'topadd_'+str(args_tmp.top)+args_tmp.name)