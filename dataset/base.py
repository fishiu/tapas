# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: base.py
@version: 1.0
@time: 2022/05/24 03:38:55
@contact: jinxy@pku.edu.cn

base dataset
"""
import logging
import pathlib

import torch.utils.data
import transformers

from .augbank import AugBank


lg = logging.getLogger()


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.data = list()
        self.aug_types = args.aug
        self.aug_bank = AugBank(self.aug_types, pathlib.Path(args.aug_dir), args)

        self.table_tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.table_model)
        lg.info(f"table tokenizer type: {type(self.table_tokenizer)}")
        self.text_tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.text_model)
        lg.info(f"text tokenizer type: {type(self.text_tokenizer)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # encode title
        title_encoding_list = []
        pos_titles = [item.title] + item.aug_titles  # add original title
        for title in pos_titles:
            title_encoding = self.text_tokenizer(
                title,
                padding="max_length",  # todo ?
                truncation=True,
                max_length=self.args.max_title_length,
                return_tensors="pt",
            )
            title_encoding = {key: value.squeeze(0) for key, value in title_encoding.items()}
            title_encoding_list.append(title_encoding)

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
            "title": title_encoding_list,
            "table": table_encoding,
        }


def collate_fn(batch):
    table_batch = [item["table"] for item in batch]
    table_batch_dict = {
        "input_ids": torch.stack([x["input_ids"] for x in table_batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in table_batch]),
        "token_type_ids": torch.stack([x["token_type_ids"] for x in table_batch]),
    }

    title_batch = []
    label_batch = []
    for i, item in enumerate(batch):
        title_list = item["title"]  # a series of positive titles
        title_batch.extend(title_list)
        label_batch.extend([i] * len(title_list))
    title_batch_dict = {
        "input_ids": torch.stack([x["input_ids"] for x in title_batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in title_batch]),
        # "token_type_ids": torch.stack([x["token_type_ids"] for x in title_batch]),
    }
    return table_batch_dict, title_batch_dict, label_batch
