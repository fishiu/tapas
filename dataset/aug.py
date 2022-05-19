# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: aug.py
@version: 1.0
@time: 2022/05/19 04:02:52
@contact: jinxy@pku.edu.cn

augmentation
"""
import pathlib
import re
import argparse
import logging
import os
import time

import torch
import torch.utils.data
import tqdm
import pandas as pd
import nlpaug
import nlpaug.augmenter.word as naw

import totto


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_dataset(args):
    dataset = totto.ToTToDataset(args.train_json, args)
    return dataset


def test_speed(aug_type: str, dataset: totto.ToTToDataset):
    print(f"test speed for {aug_type}")
    init_start_time = time.time()
    aug = init_aug(aug_type)
    init_end_time = time.time()
    init_time = init_end_time - init_start_time
    print(f"init time: {init_time}")

    run_start_time = time.time()

    # make batch
    batch_list = []
    batch = []
    batch_size = 16
    for table in dataset.data:
        batch.append(table.title)
        if len(batch) == batch_size:
            batch_list.append(batch)
            batch = []
    if len(batch) > 0:
        batch_list.append(batch)

    print(f"batch_list: {len(batch_list)}")
    for batch_text in tqdm.tqdm(batch_list):
        augmented_text = aug.augment(batch_text)
    run_end_time = time.time()
    run_time = run_end_time - run_start_time

    print(f'{aug_type}: init time {init_time}s, run time {run_time}s')


def pipeline(aug_type: str, dataset: totto.ToTToDataset, args):
    print(f"aug for {aug_type}")
    aug = init_aug(aug_type)
    print(f"augmenter init done")

    # make batch
    batch_list = []
    batch = []
    batch_size = args.batch_size
    for table in dataset.data:
        batch.append(table)
        if len(batch) == batch_size:
            batch_list.append(batch)
            batch = []
    if len(batch) > 0:
        batch_list.append(batch)
    print(f"batch_list: {len(batch_list)}")

    res_list = []
    for batch_table in tqdm.tqdm(batch_list):
        augmented_text = aug.augment([table.title for table in batch_table])
        for i, table in enumerate(batch_table):
            res_list.append({
                'id': table.id,
                'ori': table.title,
                'aug': augmented_text[i],
            })

    pd.DataFrame(res_list).to_csv(args.csv_path, index=False)


def init_aug(aug_type: str):
    if aug_type == 'ctx':
        aug_ctx = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", device='cuda:0')
        return aug_ctx
    elif aug_type == 'w2v':
        aug_w2v = naw.WordEmbsAug(model_type='glove', model_path='../data/model/glove.6B.300d.txt', action="substitute")
        return aug_w2v
    elif aug_type == 'trans':
        aug_trans = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en', device='cuda:1')
        return aug_trans
    elif aug_type == 'syno':
        aug_syno = naw.SynonymAug(aug_src='ppdb', model_path='../data/model/ppdb-2.0-s-all')
        return aug_syno


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="../data/pretrain/totto/totto_dev_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="../output/data/aug")
    parser.add_argument("--max_title_length", type=int, default=128)  # todo? table max len?
    parser.add_argument("--table_model", type=str, default="google/tapas-small")
    parser.add_argument("--text_model", type=str, default="bert-base-uncased")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--aug_type", type=str, choices=["syno", "w2v", "trans"], required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    args.output_dir = pathlib.Path(args.output_dir)
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    args.csv_path = args.output_dir / f'{args.aug_type}.csv'
    print(args)

    logging.basicConfig(level=logging.INFO)

    dataset = get_dataset(args)
    pipeline(args.aug_type, dataset, args)


if __name__ == "__main__":
    main()


"""
INFO:root:read data from ../data/pretrain/totto/totto_dev_data.jsonl
INFO:root:[('success', 1000), ('rpt set error', 110), ('row/col size > 256', 9), ('table length > 1280', 6)]

[CPU] ctx: init time 10.707417011260986s, run time 75.90963888168335s
[GPU] ctx: init time 12.934728860855103s, run time 20.898595571517944s

w2v(glove.6B.300d): init time 56.674747943878174s, run time 40.42566919326782s

[GPU] trans: init time 26.511788606643677s, run time 394.7758686542511s
[GPU-batch16] trans: init time 27.28523588180542s, run time 56.32815408706665s

syno: init time 40.769219636917114s, run time 0.4551732540130615s
"""
