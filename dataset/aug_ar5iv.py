# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: aug_ar5iv.py
@version: 1.0
@time: 2022/05/24 04:23:58
@contact: jinxy@pku.edu.cn

augment data for ar5iv
"""

import pathlib
import re
import argparse
import logging
import os

import tqdm
import pandas as pd
import nlpaug
import nlpaug.augmenter.word as naw

import ar5iv
from aug import pipeline


def init_aug(aug_type: str):
    if aug_type == 'syno':
        aug_syno = naw.SynonymAug(aug_src='ppdb', model_path='../data/model/ppdb-2.0-s-all')
        return aug_syno
    else:
        raise ValueError('aug_type must be one of [syno]')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default="../data/ar5iv_csv/", help="csv_dir")
    parser.add_argument("--aug_dir", type=str, default="../output/data/aug/ar5iv/")
    parser.add_argument("--max_title_length", type=int, default=128)
    parser.add_argument("--table_model", type=str, default="google/tapas-small")
    parser.add_argument("--text_model", type=str, default="bert-base-uncased")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--aug", type=str, nargs="*", choices=["syno"], required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.aug_dir = pathlib.Path(args.aug_dir)
    ar5iv_dataset = ar5iv.Ar5ivDataset(csv_dir=pathlib.Path(args.csv_dir), args=args)
    for aug in args.aug:
        csv_path = args.aug_dir / f"{aug}.csv"
        pipeline(aug, ar5iv_dataset, csv_path, args)


if __name__ == "__main__":
    main()