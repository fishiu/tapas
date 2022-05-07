# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: dataloader.py
@version: 1.0
@time: 2022/05/07 22:57:15
@contact: jinxy@pku.edu.cn

dataloader for table data
"""
import ast

import pandas as pd
import torch.utils.data
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer


def _parse_answer_coordinates(answer_coordinate_str):
    """ Parses the answer_coordinates of a question.

    Args:
        answer_coordinate_str: A string representation of a Python list of tuple strings.
        For example: "['(1, 4)','(1, 3)', ...]"
    """

    try:
        answer_coordinates = []
        # make a list of strings
        coords = ast.literal_eval(answer_coordinate_str)
        # parse each string as a tuple
        for row_index, column_index in sorted(ast.literal_eval(coord) for coord in coords):
            answer_coordinates.append((row_index, column_index))
    except SyntaxError:
        raise ValueError('Unable to evaluate %s' % answer_coordinate_str)

    return answer_coordinates


def _parse_answer_text(answer_text):
    """ Populates the answer_texts field of `answer` by parsing `answer_text`.

    Args:
        answer_text: A string representation of a Python list of strings.
            For example: "[u'test', u'hello', ...]"
        answer: an Answer object.
    """
    try:
        answer = []
        for value in ast.literal_eval(answer_text):
            answer.append(value)
    except SyntaxError:
        raise ValueError('Unable to evaluate %s' % answer_text)

    return answer


def get_sequence_id(example_id, annotator):
    if "-" in str(annotator):
        raise ValueError('"-" not allowed in annotator.')
    return f"{example_id}-{annotator}"


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, csv_dir, tsv_path, tokenizer, is_eval=False):
        self.csv_dir = csv_dir
        self.tokenizer = tokenizer
        self.tsv_path = tsv_path
        self.is_eval = is_eval

        data = pd.read_csv(self.tsv_path, sep="\t")
        data['answer_coordinates'] = data['answer_coordinates'].apply(
            lambda coords_str: _parse_answer_coordinates(coords_str))
        data['answer_text'] = data['answer_text'].apply(lambda txt: _parse_answer_text(txt))
        data['sequence_id'] = data.apply(lambda x: get_sequence_id(x.id, x.annotator), axis=1)

        self.df = data

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        # TapasTokenizer expects the table data to be text only
        table = pd.read_csv(self.csv_dir + item.table_file[9:]).astype(str)
        if item.position != 0:
            # use the previous table-question pair to correctly set the prev_labels token type ids
            previous_item = self.df.iloc[idx - 1]
            encoding = self.tokenizer(table=table,
                                      queries=[previous_item.question, item.question],
                                      answer_coordinates=[previous_item.answer_coordinates, item.answer_coordinates],
                                      answer_text=[previous_item.answer_text, item.answer_text],
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="pt")
            # use encodings of second table-question pair in the batch
            data_item = {key: val[-1] for key, val in encoding.items()}
        else:
            # this means it's the first table-question pair in a sequence
            encoding = self.tokenizer(table=table,
                                      queries=item.question,
                                      answer_coordinates=item.answer_coordinates,
                                      answer_text=item.answer_text,
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="pt")
            # remove the batch dimension which the tokenizer adds
            data_item = {key: val.squeeze(0) for key, val in encoding.items()}

        # add metadata for evaluation
        if self.is_eval:
            data_item['origin_encoding'] = {key: val.unsqueeze(0) for key, val in data_item.items()}
            data_item['metadata'] = {
                'id': item.id,
                'annotator': item.annotator,
                'position': item.position,
            }
        return data_item

    def __len__(self):
        return len(self.df)


def collate_fn(batch, is_eval):
    batch_dict = {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }
    if is_eval:
        batch_dict["metadata"] = [x["metadata"] for x in batch]
        batch_dict["origin_encoding"] = [x["origin_encoding"] for x in batch]
    return batch_dict
