# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: pretrain.py
@version: 1.0
@time: 2022/05/14 03:39:15
@contact: jinxy@pku.edu.cn

pretraining model and training code
"""

import os
import pathlib
import logging
import re
import json
import argparse

import torch
import torch.nn.functional
import torch.utils.data
import transformers
import info_nce
from torch.utils.tensorboard import SummaryWriter

from dataset.totto import ToTToDataset, ToTToTable, collate_fn
from utils.util import make_config


lg = logging.getLogger()


class TableCL(torch.nn.Module):
    def __init__(self, config):
        super(TableCL, self).__init__()
        # self.config = config
        self.device = config.device
        self.text_encoder = transformers.BertModel.from_pretrained(config.text_model)
        self.table_encoder = transformers.TapasModel.from_pretrained(config.table_model)

        self.text_proj = torch.nn.Linear(768, config.uni_dim)  # unified_dim = 512
        self.table_proj = torch.nn.Linear(512, config.uni_dim)  # unified_dim = 512

        self.criterion = info_nce.InfoNCE()

    def forward(self, table_inputs, text_inputs):
        """forward both table and text input, get InfoNCE loss

        Args:
            table_inputs: [batch_size, table_seq_len]
            text_inputs: [batch_size, text_seq_len]

        Returns:
            InfoNCE loss
        """
        table_encoded = self.table_encoder(
            input_ids=table_inputs["input_ids"].to(self.device),
            attention_mask=table_inputs["attention_mask"].to(self.device),
            token_type_ids=table_inputs["token_type_ids"].to(self.device),
        )
        table_embedded = self.table_proj(table_encoded.pooler_output)

        text_encoded = self.text_encoder(
            input_ids=text_inputs["input_ids"].to(self.device),
            attention_mask=text_inputs["attention_mask"].to(self.device),
            token_type_ids=text_inputs["token_type_ids"].to(self.device),
        )
        text_embedded = self.text_proj(text_encoded.pooler_output)

        # table as query list, text as paired
        loss = self.criterion(query=table_embedded, positive_key=text_embedded)  # todo better debug on this loss
        return loss


def train(model: TableCL, optimizer, dataloader, args):
    tb = SummaryWriter(args.tensorboard_dir)
    model.train()
    total_step = 1

    for epoch in range(1, args.epochs + 1):  # start from 1
        report_loss = 0.
        for step, batch in enumerate(dataloader):
            table_inputs, title_inputs = batch
            loss = model(table_inputs, title_inputs)
            report_loss += loss.item()
            tb.add_scalar("loss", loss.item(), total_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if total_step % args.report_step == 0:
                report_loss /= args.report_step
                lg.info(f"[TRAIN] epoch: {epoch}, step: {step}/{len(dataloader)}, loss: {report_loss:.4f}")
                save_name = f"{epoch}_{total_step}_{report_loss:.4f}"
                report_loss = 0.
                if total_step % args.save_step == 0:
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{save_name}.pth"))
                    torch.save(optimizer.state_dict(), os.path.join(args.checkpoint_dir, f"{save_name}.opt"))
                    lg.info(f"[SAVE] save model to {save_name}")
            total_step += 1


def get_parser():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--train_json", type=str, default="data/pretrain/totto/totto_train_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/pretrain/0_demo")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--report_step", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=1000)

    # data
    parser.add_argument("--max_title_length", type=int, default=128)  # todo? table max len?

    # huggingface
    # parser.add_argument("--table_encoder", type=str, default="google/tapas-base")
    # parser.add_argument("--table_tokenizer", type=str, default="google/tapas-base")
    # parser.add_argument("--text_encoder", type=str, default="bert-base-uncased")
    # parser.add_argument("--text_tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--table_model", type=str, default="google/tapas-small")
    parser.add_argument("--text_model", type=str, default="bert-base-uncased")

    # model
    parser.add_argument("--uni_dim", type=int, default=512, help="projection dim for both modality")

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1107)

    parser.add_argument("--debug", action="store_true")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    make_config(args)
    assert args.save_step % args.report_step == 0, "save_step should be multiple of report_step"
    lg.info("=" * 50)
    lg.info(args)

    train_dataset = ToTToDataset(args.train_json, args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=args.shuffle,
                                                   collate_fn=collate_fn,
                                                   num_workers=8)

    model = TableCL(args)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, train_dataloader, args)


def debug_table(json_path, args):
    with open(json_path, 'r') as f:
        table_data = json.load(f)
    table = ToTToTable(table_data, args)
    tokenizer = transformers.TapasTokenizer.from_pretrained(args.table_tokenizer)
    table_encoding = tokenizer(
        table=table.table_df,
        queries=[""],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


if __name__ == "__main__":
    main()
    # debug_table("data/pretrain/totto/sample.json", get_parser().parse_args())
