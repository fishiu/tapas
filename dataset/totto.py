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

import pandas as pd
import torch.utils.data
import transformers


lg = logging.getLogger()


class ToTToException(Exception):
    ...


class ToTToTable:
    """read json data and convert to TAPAS table format"""
    def __init__(self, json_data, config):
        self.title = self.__merge_title(json_data["table_page_title"], json_data["table_section_title"])
        self.table_df = self.__read_table(json_data["table"])

    @staticmethod
    def __merge_title(page_title, section_title):
        if len(page_title) + len(section_title) < 3:
            raise ToTToException("title length < 3")
        if len(page_title) + len(section_title) > 256:
            raise ToTToException("title length > 256")
        return page_title + '. ' + section_title + '.'

    @staticmethod
    def __read_table(table_data):
        table = []
        row_len_set = set()
        for row_data in table_data:
            row = []
            for cell_data in row_data:
                row_span = cell_data["row_span"]
                col_span = cell_data["column_span"]
                cell_value = cell_data["value"]
                row.extend([cell_value for _ in range(col_span)])
                if row_span > 1:
                    # currently we only support row_span = 1
                    raise ToTToException("row_span > 1")
            row_len_set.add(len(row))
            table.append(row)
        if len(row_len_set) != 1:
            raise ToTToException("row_len_set != 1")
        row_size, col_size = len(table), len(table[0])
        if col_size > 256 or row_size > 256:
            raise ToTToException("row/col size > 256")
        if col_size * row_size > 1280:
            raise ToTToException("table length > 1280")
        table = pd.DataFrame(table[1:], columns=table[0]).astype(str)
        return table


class ToTToDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, args):
        self.args = args
        self.json_path = json_path
        self.data = self.__read_data()
        self.table_tokenizer = transformers.TapasTokenizer.from_pretrained(self.args.table_model)
        self.text_tokenizer = transformers.BertTokenizer.from_pretrained(self.args.text_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """todo make sure the data size"""
        item = self.data[idx]
        # encode title
        title_encoding = self.text_tokenizer(
            item.title,
            padding="max_length",  # todo ?
            truncation=True,
            max_length=self.args.max_title_length,
            return_tensors="pt",
        )
        title_encoding = {key: value.squeeze(0) for key, value in title_encoding.items()}

        # encode table
        table_encoding = self.table_tokenizer(
            table=item.table_df,
            queries=[""],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )  # todo need debugging
        table_encoding = {key: value.squeeze(0) for key, value in table_encoding.items()}

        return {
            "title": title_encoding,
            "table": table_encoding,
        }

    def __read_data(self):
        lg.info(f"read data from {self.json_path}")
        data = list()
        stat = list()
        debug_cnt = 0
        with open(self.json_path, 'r') as f:
            while not self.args.debug or (self.args.debug and debug_cnt < 1000):
                # read json data for one table
                totto_data_str = f.readline().strip()
                if not totto_data_str: break
                totto_data = json.loads(totto_data_str)

                try:
                    table = ToTToTable(totto_data, self.args)
                except ToTToException as e:
                    stat.append(str(e))
                    continue
                else:
                    stat.append("success")
                    data.append(table)
                    debug_cnt += 1
        lg.info(collections.Counter(stat).most_common())
        return data


def collate_fn(table_batch):
    table_batch_dict = {
        "input_ids": torch.stack([x["table"]["input_ids"] for x in table_batch]),
        "attention_mask": torch.stack([x["table"]["attention_mask"] for x in table_batch]),
        "token_type_ids": torch.stack([x["table"]["token_type_ids"] for x in table_batch]),
    }
    title_batch_dict = {
        "input_ids": torch.stack([x["title"]["input_ids"] for x in table_batch]),
        "attention_mask": torch.stack([x["title"]["attention_mask"] for x in table_batch]),
        "token_type_ids": torch.stack([x["title"]["token_type_ids"] for x in table_batch]),
    }
    return table_batch_dict, title_batch_dict
