# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: augbank.py
@version: 1.0
@time: 2022/05/24 03:31:46
@contact: jinxy@pku.edu.cn

aug bank
"""

import pathlib
import logging

import pandas as pd


lg = logging.getLogger()


class AugBank:
    def __init__(self, aug_types: list, aug_data_dir: pathlib.Path, args):
        self.dataset = args.dataset
        self.aug_types = aug_types
        self.aug_data_dir = aug_data_dir
        self.aug_data = dict()
        self._load_aug_data()

    def _load_aug_data(self):
        for aug_type in self.aug_types:
            aug_data_path = self.aug_data_dir / f"{aug_type}.csv"
            if not aug_data_path.exists():
                lg.error(f"aug data not exists: {aug_data_path}")
                continue
            aug_data = pd.read_csv(aug_data_path, index_col=0)
            len_before_dup = len(aug_data)
            aug_data = aug_data[~aug_data.index.duplicated(keep='first')]
            len_after_dup = len(aug_data)
            lg.warning(f"data has {len_before_dup} samples, after dedup {len_after_dup} samples")
            self.aug_data[aug_type] = aug_data.to_dict("index")
            lg.info(f"load aug data {aug_type} from {aug_data_path}")

    def get_aug_sample(self, table_id, aug_type):
        if aug_type not in self.aug_types:
            raise NameError(f"aug_type {aug_type} not in {self.aug_types}")
        if table_id not in self.aug_data[aug_type]:
            return
        data = self.aug_data[aug_type][table_id]
        ori_title = data['ori']
        aug_title = data['aug']
        if ori_title == aug_title:
            return
        return aug_title  # debug, return only aug later
