# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: ar5iv.py
@version: 1.0
@time: 2022/05/24 03:23:19
@contact: jinxy@pku.edu.cn

arxiv dataset
"""
import collections
import logging
import pathlib

import pandas as pd

from .base import BaseDataset

lg = logging.getLogger()


class Ar5ivException(Exception):
    ...


class Ar5ivTable:
    def __init__(self, csv_dir, csv_path, title):
        csv_path = csv_path.replace("../data/ar5iv_csv/", "")
        self.csv_path = csv_dir / csv_path
        self.id = pathlib.Path(csv_path).stem
        self.title = title
        title_len = len(self.title.split())
        if title_len > 128:
            raise Ar5ivException("title length > 128")
        if title_len < 5:
            raise Ar5ivException("title length < 5")
        self.table_df = self._read_table()
        row_size, col_size = self.table_df.shape
        if col_size > 256 or row_size > 256:
            raise Ar5ivException("row/col size > 256")
        if col_size * row_size > 1280:
            raise Ar5ivException("table length > 1280")
        self.aug_titles = []

    def _read_table(self):
        # astype(str) is very important
        return pd.read_csv(self.csv_path).astype(str)

    def __repr__(self):
        return self.title


class Ar5ivDataset(BaseDataset):
    def __init__(self, csv_dir: pathlib.Path, args):
        super(Ar5ivDataset, self).__init__(args)
        self.csv_dir = csv_dir
        self.stat = list()
        self.aug_stat = {aug_type: 0 for aug_type in self.aug_types}
        for d in csv_dir.iterdir():
            if not d.is_dir():
                lg.warning(f"{d} is not a directory")
                continue
            self.meta_path = d / "meta_title.csv"
            self._read_data(self.meta_path)
            if args.debug:
                break
        lg.info(collections.Counter(self.stat).most_common())
        """
        [('success', 23197), ('title length < 5', 2765), ('title length > 128', 107)]
        """
        lg.info(f"aug stat: {self.aug_stat}")

    def _read_data(self, meta_path):
        meta_df = pd.read_csv(meta_path).astype(str)
        lg.info(f"Reading meta data from {self.meta_path}, data length: {len(meta_df)}")
        for _, row in meta_df.iterrows():
            # csv_dir is not needed, because csv_path is absolute path
            # caption is with "Table X: ", title is pure title
            try:
                table = Ar5ivTable(self.csv_dir, row["csv_path"], row["title"])
            except Ar5ivException as e:
                self.stat.append(str(e))
                continue
            else:
                for aug_type in self.aug_types:
                    aug_title = self.aug_bank.get_aug_sample(table.id, aug_type)
                    if aug_title:
                        table.aug_titles.append(aug_title)
                        self.aug_stat[aug_type] += 1
                self.stat.append("success")
                self.data.append(table)
