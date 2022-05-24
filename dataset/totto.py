# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: totto.py
@version: 1.0
@time: 2022/05/14 17:19:50
@contact: jinxy@pku.edu.cn

totto dataset
"""
import json
import collections
import logging
import pathlib

import pandas as pd

from .base import BaseDataset

lg = logging.getLogger()


class ToTToException(Exception):
    ...


class ToTToTable:
    """read json data and convert to TAPAS table format"""
    def __init__(self, json_data):
        self.title = self._merge_title(json_data["table_page_title"], json_data["table_section_title"])
        self.id = json_data["example_id"]
        self.table_df = self._read_table(json_data["table"])
        self.aug_titles = []

    def __repr__(self):
        return self.title

    @staticmethod
    def _merge_title(page_title, section_title):
        page_title_len = len(page_title.split(" "))
        section_title_len = len(section_title.split(" "))
        if page_title_len + section_title_len < 5:
            raise ToTToException("title length < 5")
        if page_title_len + section_title_len > 128:
            raise ToTToException("title length > 128")
        return page_title + '. ' + section_title + '.'

    @staticmethod
    def _read_table(table_data):
        table = []
        vert_merge = {}  # vertical_merge_info_dict: start_pos, len, depth, value
        rpt_set = set()
        for row_data in table_data:
            row = []
            rpt = 0  # row pointer
            for cell_data in row_data:
                while rpt in vert_merge:  # merge on this column
                    info = vert_merge[rpt]
                    row.extend([info["value"] for _ in range(info["len"])])  # fill row
                    tmp = rpt
                    rpt += info["len"]  # go forward
                    info["depth"] -= 1
                    if info["depth"] == 0:  # no remain depth to go
                        del vert_merge[tmp]

                # get basic info
                row_span = cell_data["row_span"]
                col_span = cell_data["column_span"]
                cell_value = cell_data["value"]
                # push into row
                row.extend([cell_value for _ in range(col_span)])

                if row_span > 1:  # found a vertical merge
                    assert rpt not in vert_merge, "rpt already exist in vert_merge"
                    vert_merge[rpt] = {
                        "value": cell_value,
                        "len": col_span,
                        "depth": row_span - 1,  # remaining
                    }
                rpt += col_span

            # do not forget to check after the cell loop
            while rpt in vert_merge:  # merge on this column
                info = vert_merge[rpt]
                row.extend([info["value"] for _ in range(info["len"])])  # fill row
                tmp = rpt
                rpt += info["len"]  # go forward
                info["depth"] -= 1
                if info["depth"] == 0:  # no remain depth to go
                    del vert_merge[tmp]

            # finally get a row, rpt should be grid width
            table.append(row)
            rpt_set.add(rpt)

        row_size, col_size = len(table), len(table[0])
        if len(rpt_set) != 1:
            raise ToTToException("rpt set error")
        if col_size > 256 or row_size > 256:
            raise ToTToException("row/col size > 256")
        if col_size * row_size > 1280:
            raise ToTToException("table length > 1280")
        table = pd.DataFrame(table[1:], columns=table[0]).astype(str)
        return table


class ToTToDataset(BaseDataset):
    def __init__(self, json_path, args):
        super(ToTToDataset, self).__init__(args)
        self.json_path = json_path
        self.data = self._read_data()

    def _read_data(self):
        lg.info(f"read data from {self.json_path}")
        data = list()
        stat = list()
        aug_stat = {aug_type: 0 for aug_type in self.aug_types}
        debug_cnt = 0
        with open(self.json_path, 'r') as f:
            while not self.args.debug or (self.args.debug and debug_cnt < 100):
                # read json data for one table
                totto_data_str = f.readline().strip()
                if not totto_data_str: break
                totto_data = json.loads(totto_data_str)

                try:
                    table = ToTToTable(totto_data)
                except ToTToException as e:
                    stat.append(str(e))
                    continue
                else:
                    # aug title
                    for aug_type in self.aug_types:
                        aug_title = self.aug_bank.get_aug_sample(table.id, aug_type)
                        if aug_title:
                            table.aug_titles.append(aug_title)
                            aug_stat[aug_type] += 1
                    stat.append("success")
                    data.append(table)
                    debug_cnt += 1
        lg.info(f"table stat: {collections.Counter(stat).most_common()}")
        lg.info(f"aug stat: {aug_stat}")
        return data


def merge_data():
    """merge train, dev, test"""
    train_path = pathlib.Path("../data/pretrain/totto/totto_train_data.jsonl")  # 120761
    dev_path = pathlib.Path("../data/pretrain/totto/totto_dev_data.jsonl")  # 7700
    test_path = pathlib.Path("../data/pretrain/totto/unlabeled_totto_test_data.jsonl")  # 7700
    merge_path = pathlib.Path("../data/pretrain/totto/totto_merge_data.jsonl")
    with merge_path.open('w') as f:
        for path in [train_path, dev_path, test_path]:
            cnt = 0
            with path.open('r') as f_:
                for line in f_:
                    f.write(line)
                    cnt += 1
            print(f"{path} has {cnt} lines")


if __name__ == "__main__":
    merge_data()
