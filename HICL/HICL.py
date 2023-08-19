import argparse
import logging
# from transformers.utils import logging
import os
import sys
import random
import time
import math
import distutils.util
from functools import partial
import copy
import numpy as np
import datasets
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from tqdm import trange,tqdm

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

CONVERT = {
    'eval-emotion':{'0':0,'1':1,'2':2,'3':3},
    'eval-hate':{'0':0,'1':1},
    'eval-irony':{'0':0,'1':1},
    'eval-offensive':{'0':0,'1':1},
    'eval-stance': {'0': 0, '1': 1, '2': 2},
    'sem22-task6-sarcasm': {'0': 0, '1': 1},
    'sem21-task7-humor': {'0': 0, '1': 1}
}
from transformers import RobertaForSequenceClassification
import torch.nn as nn
class RobertaForMulti(RobertaForSequenceClassification):
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        num_old = self.roberta.config.max_position_embeddings
        if num_old >= new_num_position_embeddings:
            return
        self.roberta.config.max_position_embeddings = new_num_position_embeddings
        # old_position_embeddings_weight = self.roberta.embeddings.position_embeddings.weight.clone()
        new_position = nn.Embedding(self.roberta.config.max_position_embeddings, self.roberta.config.hidden_size)
        new_position.to(self.roberta.embeddings.position_embeddings.weight.device,
                        dtype=self.roberta.embeddings.position_embeddings.weight.dtype)
        # self._init_weights(new_position)
        new_position.weight.data[:num_old, :] = self.roberta.embeddings.position_embeddings.weight.data[:num_old, :]
        self.roberta.embeddings.position_embeddings = new_position

        self.roberta.embeddings.register_buffer("position_ids",
                                                torch.arange(self.roberta.config.max_position_embeddings).expand(
                                                    (1, -1)))
        self.roberta.embeddings.register_buffer(
            "token_type_ids", torch.zeros([1, self.roberta.config.max_position_embeddings], dtype=torch.long),
            persistent=False
        )

from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Tuple
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
@dataclass
class OurDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    template: Optional[list] = None

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[
        str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for idx_fea in range(len(features)):
            feature = features[idx_fea]
            flat_features.append({k: feature[k][0] if k in special_keys else feature[k] for k in feature})
            if self.template is None:
                flat_features[idx_fea]['input_ids'] = sum(feature['input_ids'][1:],flat_features[idx_fea]['input_ids'])
            else:
                flat_features[idx_fea]['input_ids'] = sum(feature['input_ids'][1:], self.template[0] + flat_features[idx_fea]['input_ids'] + self.template[1]) + self.template[2]
            flat_features[idx_fea]['attention_mask'] = flat_features[idx_fea]['attention_mask'] + \
                                                       [1]*(len(flat_features[idx_fea]['input_ids']) - len(flat_features[idx_fea]['attention_mask']))
            flat_features[idx_fea]['token_type_ids'] = flat_features[idx_fea]['token_type_ids'] + \
                                                       [self.tokenizer.pad_token_type_id] * (len(flat_features[idx_fea]['input_ids']) - len(flat_features[idx_fea]['token_type_ids']))

            # for i in range(num_sent):
                # flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
        #          for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]

        return batch




def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm',
        type=str,
        required=False,
        help="The name of the task to train")
    parser.add_argument(
        "--model_name_or_path",
        default='vinai/bertweet-base',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: "
    )
    parser.add_argument(
        "--token_name_or_path",
        default='vinai/bertweet-base',
        type=str,
        required=False,
    )
    parser.add_argument(
        "--input_dir",
        default='../data/',
        type=str,
        required=False,
        help="The input directory where the data are stored.",
    )
    parser.add_argument(
        "--method",
        default='hash_hicl_topadd_1',
        type=str,
        required=False,
        help="Token data save directory",
    )
    parser.add_argument(
        "--results_name",
        default='results_ft_retrione20.txt',
        type=str,
        required=False,
        help="Save names for the results.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=200,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--token_type",
        default=1,
        type=int)
    parser.add_argument(
        "--learning_rate",
        default='1e-5',#'1e-3,1e-4,1e-5,1e-6',
        type=str,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=30,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant_with_warmup",
        help="The scheduler type to use.",
        # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0.1, help="Ratio of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps",
        default=None,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default='0,1,2,3,4,5,6,7,8,9', type=str, help="random seed for initialization")
    parser.add_argument(
        "--shot", default='full', type=str, help="few shot")
    parser.add_argument(
        "--stop", default=5, type=int, help="early stop")
    parser.add_argument(
        "--weight", default=0, type=int, help="weighted loss")
    parser.add_argument(
        "--write_result", default='', type=str, help="weighted loss")
    parser.add_argument(
        "--template", default='050', type=str, help="number and place of trigger words")
    parser.add_argument(
        "--save_model", default='', type=str, help="save model")
    args = parser.parse_args()
    return args

@torch.no_grad()
def evaluate(model, data_loader, task='eval-emoji',write_result=''):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        # input_ids, segment_ids, labels = batch
        # logits = model(input_ids.cuda(), segment_ids.cuda())
        logits = model(input_ids=batch['input_ids'].cuda(),
                       # token_type_ids=batch['token_type_ids'].cuda(),
                       attention_mask=batch['attention_mask'].cuda()).logits
        labels = batch['labels']
        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]
    if len(write_result) > 0:
        with open(write_result, 'a', encoding='utf-8') as f:
            f.write(task+'\n')
            for one in pred_all:
                f.write(str(one))
            f.write('\n')
    results = classification_report(label_all, pred_all, output_dict=True)

        # Emotion (Macro f1)
    if 'emotion' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Hate (Macro f1)
    elif 'hate' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Irony (Irony class f1)
    elif 'irony' in task:
        tweeteval_result = results['1']['f1-score']

        # Offensive (Macro f1)
    elif 'offensive' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Stance (Macro F1 of 'favor' and 'against' classes)
    elif 'stance' in task:
        f1_against = results['1']['f1-score']
        f1_favor = results['2']['f1-score']
        tweeteval_result = (f1_against + f1_favor) / 2
    elif 'sarcasm' in task:
        tweeteval_result = results['1']['f1-score']
    elif 'humor' in task:
        tweeteval_result = results['1']['f1-score']

    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (tweeteval_result, tweeteval_result, tweeteval_result))
    return tweeteval_result,tweeteval_result,tweeteval_result

def convert_example(example, label2idx):
    if example.get('special_tokens_mask') is not None:
        example.pop('special_tokens_mask')
    example['labels'] = label2idx[example['labels']]
    return example  # ['input_ids'], example['token_type_ids'], label, prob

MAX_NUM_VECTORS = 50

def get_new_token(vid):
    assert(vid >= 0 and vid < MAX_NUM_VECTORS)
    return '[C_P_%d]'%(vid)

def prepare_for_dense_prompt(model, tokenizer):
    new_tokens = [get_new_token(i) for i in range(MAX_NUM_VECTORS)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

def get_template_text(manual_template, tokenizer):
    new_token_id = 0
    template = []
    for idx in range(len(manual_template)):
        template.append([])
        word = manual_template[idx]
        if word.isdigit():
            tokens = int(word)
            for token in range(tokens):
                tmp = tokenizer.convert_tokens_to_ids(get_new_token(new_token_id))
                template[idx].append(tmp)
                new_token_id += 1
        else:
            raise ValueError('invalid template input')
    return template

def do_train(args):
    print(args)
    data_all = datasets.load_from_disk(args.input_dir)
    label2idx = CONVERT[args.task.split('_')[0]]
    trans_func = partial(
        convert_example,
        label2idx=label2idx)
    train_ds = data_all['train']
    train_ds = train_ds.map(trans_func)
    if len(args.shot) > 0:
        if args.shot != 'full':
            sample_num = int(args.shot)
            train_ds = train_ds.shuffle()
            select_idx = []
            select_idx_dic = {}
            for val in label2idx.values():
                select_idx_dic[val] = 0
            for idx in range(len(train_ds)):
                label_tmp = train_ds[idx]['labels']
                if select_idx_dic[label_tmp] < sample_num:
                    select_idx.append(idx)
                    select_idx_dic[label_tmp] += 1
            np.random.shuffle(select_idx)
            train_ds = train_ds.select(select_idx)

    dev_ds = data_all['dev']
    dev_ds = dev_ds.map(trans_func)
    test_ds = data_all['test']
    test_ds = test_ds.map(trans_func)

    learning_rate = args.learning_rate.split(',')
    best_metric = [0, 0, 0]
    model_best = None
    for lr in learning_rate:
        best_metric_lr = [0, 0, 0]
        num_classes = len(label2idx.keys())
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_classes)
        if 'bertweet' in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
            tokenizer.model_max_length = args.max_seq_length
            model = RobertaForMulti.from_pretrained(
                args.model_name_or_path, config=config)
            model.resize_position_embeddings(args.max_seq_length)
            model = model.cuda()
            # model.resize_type_embeddings(args.token_type)
        elif 'roberta' in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            tokenizer.model_max_length = args.max_seq_length
            model = RobertaForMulti.from_pretrained(
                args.model_name_or_path, config=config)
            model.resize_position_embeddings(args.max_seq_length)
            model = model.cuda()
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config).cuda()
        tokenizer._pad_token_type_id = args.token_type - 1

        # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # new_tokens = [get_new_token(i) for i in range(MAX_NUM_VECTORS)]
        # tokenizer.add_tokens(new_tokens)
        original_vocab_size = len(list(tokenizer.get_vocab()))
        prepare_for_dense_prompt(model,tokenizer)
        template = get_template_text(args.template.strip(), tokenizer)

        batchify_fn = OurDataCollatorWithPadding(tokenizer=tokenizer,template=template)
        train_data_loader = DataLoader(
            train_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        dev_data_loader = DataLoader(
            dev_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        test_data_loader = DataLoader(
            test_ds, shuffle=False, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        print('data ready!!!')
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(lr))
        num_update_steps_per_epoch = len(train_data_loader)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(args.num_warmup_steps*args.max_train_steps),
            num_training_steps=args.max_train_steps,
        )

        loss_fct = nn.CrossEntropyLoss().cuda()
        if args.weight == 1:# or 'sarcasm' in args.task:
            num_dic = {}
            for val in label2idx.values():
                num_dic[val] = 0.0
            for idx in range(len(train_ds)):
                label_tmp = train_ds[idx]['labels']
                num_dic[label_tmp] += 1.0
            num_max = max(num_dic.values())
            class_weights = [num_max / i for i in num_dic.values()]
            class_weights = torch.FloatTensor(class_weights).cuda()
            loss_fct = nn.CrossEntropyLoss(weight=class_weights).cuda()

        print('start Training!!!')
        global_step = 0
        tic_train = time.time()

        stop_sign = 0
        for epoch in trange(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                # print(batch['input_ids'].shape)
                # input_ids, segment_ids, labels = batch
                # logits = model(input_ids.cuda(), segment_ids.cuda())
                # loss = loss_fct(logits, labels.cuda().view(-1))
                logits = model(input_ids=batch['input_ids'].cuda(),
                               # token_type_ids = batch['token_type_ids'].cuda(),
                               attention_mask=batch['attention_mask'].cuda() ).logits
                loss = loss_fct(logits, batch['labels'].cuda().view(-1))
                # print(step)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if (epoch + 1) % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, loss: %f, speed: %.4f step/s, seed: %d,lr: %.5f,task: %s"
                    % (global_step, args.max_train_steps, epoch,
                       loss, args.logging_steps / (time.time() - tic_train),
                       args.seed,float(lr),args.input_dir))
                tic_train = time.time()
            if (epoch + 1) % args.save_steps == 0 and (epoch + 1) > 3:
                tic_eval = time.time()
                cur_metric = evaluate(model, dev_data_loader, args.task)
                print("eval done total : %s s" % (time.time() - tic_eval))
                if cur_metric[0] > best_metric_lr[0]:
                    best_metric_lr = cur_metric
                    stop_sign = 0
                    if best_metric_lr[0] > best_metric[0]:
                        model_best = copy.deepcopy(model).cpu()
                        best_metric = best_metric_lr

                else:
                    stop_sign += 1
            if stop_sign >= args.stop:
                break
        del model
        torch.cuda.empty_cache()
    if model_best is None:
        cur_metric = [0.0,0.0,0.0]
        return cur_metric
    else:
        model = model_best.cuda()

    ##############update triggers
    learning_rate = args.learning_rate.split(',')
    # best_metric = [0, 0, 0]
    # model_best = None
    for lr in learning_rate:
        best_metric_lr = [0, 0, 0]

        no_decay = ["bias", "LayerNorm.weight"]
        if 'bart' in args.model_name_or_path:
            optimizer = AdamW([{'params': model.model.decoder.embed_tokens.parameters()}
                               # {'params': model.model.encoder.embed_tokens.parameters()}
                               ],
                              lr=float(lr),correct_bias=False)
        else:
            optimizer = AdamW([{'params': model.roberta.embeddings.word_embeddings.parameters()}], lr=float(lr),
                              correct_bias=False)
        num_update_steps_per_epoch = len(train_data_loader)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        loss_fct = nn.CrossEntropyLoss().cuda()
        if args.weight == 1:# or 'sarcasm' in args.task:
            num_dic = {}
            for val in label2idx.values():
                num_dic[val] = 0.0
            for idx in range(len(train_ds)):
                label_tmp = train_ds[idx]['labels']
                num_dic[label_tmp] += 1.0
            num_max = max(num_dic.values())
            class_weights = [num_max / i for i in num_dic.values()]
            class_weights = torch.FloatTensor(class_weights).cuda()
            loss_fct = nn.CrossEntropyLoss(weight=class_weights).cuda()

        print('start Training!!!')
        global_step = 0
        tic_train = time.time()

        stop_sign = 0
        for epoch in trange(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                # print(batch['input_ids'].shape)
                # input_ids, segment_ids, labels = batch
                # logits = model(input_ids.cuda(), segment_ids.cuda())
                # loss = loss_fct(logits, labels.cuda().view(-1))
                logits = model(input_ids=batch['input_ids'].cuda(),
                               # token_type_ids = batch['token_type_ids'].cuda(),
                               attention_mask=batch['attention_mask'].cuda() ).logits
                loss = loss_fct(logits, batch['labels'].cuda().view(-1))
                # print(step)
                loss.backward()
                if 'bart' in args.model_name_or_path:
                    for p in model.model.decoder.embed_tokens.parameters():
                        p.grad[:original_vocab_size, :] = 0.0
                    for p in model.model.encoder.embed_tokens.parameters():
                        p.grad[:original_vocab_size, :] = 0.0
                else:
                    for p in model.roberta.embeddings.word_embeddings.parameters():
                        # only update new tokens
                        p.grad[:original_vocab_size, :] = 0.0

                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()
            if (epoch + 1) % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, loss: %f, speed: %.4f step/s, seed: %d,lr: %.5f,task: %s"
                    % (global_step, args.max_train_steps, epoch,
                       loss, args.logging_steps / (time.time() - tic_train),
                       args.seed,float(lr),args.input_dir))
                tic_train = time.time()
            if (epoch + 1) % args.save_steps == 0:
                tic_eval = time.time()
                cur_metric = evaluate(model, dev_data_loader, args.task)
                print("trigger eval done total : %s s" % (time.time() - tic_eval))
                if cur_metric[0] > best_metric_lr[0]:
                    best_metric_lr = cur_metric
                    stop_sign = 0
                    if best_metric_lr[0] > best_metric[0]:
                        model_best = copy.deepcopy(model).cpu()
                        best_metric = best_metric_lr

                else:
                    stop_sign += 1
            if stop_sign >= args.stop:
                break
        del model
        torch.cuda.empty_cache()
    if model_best is None:
        cur_metric = [0.0,0.0,0.0]
    else:
        model = model_best.cuda()
        cur_metric = evaluate(model, test_data_loader,args.task,args.write_result)
        if args.save_model != '':
            model.save_pretrained('./'+args.save_model+'/'+args.task+'/'+str(args.seed))
        del model

    print('final')
    print("f1macro:%.5f, acc:%.5f, acc: %.5f, " % (best_metric[0], best_metric[1], best_metric[2]))
    print("f1macro:%.5f, acc:%.5f, acc: %.5f " % (cur_metric[0], cur_metric[1], cur_metric[2]))

    return cur_metric

if __name__ == "__main__":
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # r_dir = '/work/test/finetune/continue/'
    for shot in args.shot.split(','):
        for task in args.task_name.split(','):
            for model_name in args.model_name_or_path.split(','):  # [r_dir+'bertweet/']:
                ave_metric = []
                for seed in args.seed.split(','):
                    set_seed(int(seed))
                    args_tmp = copy.deepcopy(args)
                    args_tmp.task = task
                    args_tmp.input_dir = args.input_dir + task + '/' + args.method
                    args_tmp.seed = int(seed)
                    args_tmp.shot = shot
                    args_tmp.model_name_or_path = model_name
                    ave_metric.append(do_train(args_tmp))
                ave_metric = np.array(ave_metric)
                num_seed = len(args.seed.split(','))
                print("*************************************************************************************")
                print('Task: %s, model: %s, shot: %s' % (task, model_name, shot))
                print('final aveRec:%.5f, f1PN:%.5f, acc: %.5f ' % (sum(ave_metric[:, 0]) / num_seed,
                                                                    sum(ave_metric[:, 1]) / num_seed,
                                                                    sum(ave_metric[:, 2]) / num_seed))
                with open(args.results_name, 'a') as f_res:

                    f_res.write('Task: %s, model: %s, shot: %s\n' % (task, model_name, shot) )
                    f_res.write('aveRec:%.5f, f1PN:%.5f, acc: %.5f \n' % (sum(ave_metric[:, 0]) / num_seed,
                                                                          sum(ave_metric[:, 1]) / num_seed,
                                                                          sum(ave_metric[:, 2]) / num_seed))
                    for tmp in range(num_seed):
                        f_res.write('%.5f, %.5f, %.5f \n' % (ave_metric[tmp, 0],ave_metric[tmp, 1],ave_metric[tmp, 2]))

                    f_res.close()
