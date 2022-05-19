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
import torch.utils.data
import transformers


lg = logging.getLogger()


class ToTToException(Exception):
    ...


class AugBank:
    def __init__(self, aug_types: list, aug_data_dir: pathlib.Path):
        self.aug_types = aug_types
        self.aug_data_dir = aug_data_dir
        self.aug_data = dict()
        self._load_aug_data()

    def _load_aug_data(self):
        for aug_type in self.aug_types:
            aug_data_path = self.aug_data_dir / f"{aug_type}.csv"
            aug_data = pd.read_csv(aug_data_path, index_col=0)
            aug_data = aug_data[~aug_data.index.duplicated(keep='first')]
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


class ToTToTable:
    """read json data and convert to TAPAS table format"""
    def __init__(self, json_data, config):
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
            raise ToTToException("title length > 256")
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


class ToTToDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, args):
        self.args = args
        self.aug_types = args.aug
        self.json_path = json_path
        self.aug_bank = AugBank(self.aug_types, pathlib.Path(args.aug_dir))
        self.data = self._read_data()

        self.table_tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.table_model)
        lg.info(f"table tokenizer type: {type(self.table_tokenizer)}")
        self.text_tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.text_model)
        lg.info(f"text tokenizer type: {type(self.text_tokenizer)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """todo make sure the data size"""
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

    def _read_data(self):
        lg.info(f"read data from {self.json_path}")
        data = list()
        stat = list()
        aug_stat = list()
        debug_cnt = 0
        with open(self.json_path, 'r') as f:
            while not self.args.debug or (self.args.debug and debug_cnt < 100):
                # read json data for one table
                totto_data_str = f.readline().strip()
                if not totto_data_str: break
                totto_data = json.loads(totto_data_str)

                try:
                    table = ToTToTable(totto_data, self.args)
                    # aug title
                    for aug_type in self.aug_types:
                        aug_title = self.aug_bank.get_aug_sample(table.id, aug_type)
                        if aug_title:
                            table.aug_titles.append(aug_title)
                        aug_stat.append(len(table.aug_titles))
                except ToTToException as e:
                    stat.append(str(e))
                    continue
                else:
                    stat.append("success")
                    data.append(table)
                    debug_cnt += 1
        lg.info(collections.Counter(stat).most_common())
        lg.info(collections.Counter(aug_stat).most_common())
        return data


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
