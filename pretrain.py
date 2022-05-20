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
from torch.utils.tensorboard import SummaryWriter

from dataset.totto import ToTToDataset, ToTToTable, collate_fn
from utils.util import make_config
from utils.info_nce import InfoNCE


lg = logging.getLogger()


class TableCL(torch.nn.Module):
    def __init__(self, config):
        super(TableCL, self).__init__()
        # self.config = config
        self.device = config.device
        self.text_encoder = transformers.AutoModel.from_pretrained(config.text_model)
        lg.info(f"text encoder type: {type(self.text_encoder)}")
        self.table_encoder = transformers.AutoModel.from_pretrained(config.table_model)
        lg.info(f"table encoder type: {type(self.table_encoder)}")

        self.text_proj = torch.nn.Linear(config.text_hidden_dim, config.uni_dim)  # unified_dim = 512
        self.table_proj = torch.nn.Linear(config.table_hidden_dim, config.uni_dim)  # unified_dim = 512

        self.criterion = InfoNCE()

    def forward(self, table_inputs, text_inputs, labels):
        """forward both table and text input, get InfoNCE loss

        Args:
            table_inputs: [table_batch_size, table_seq_len]
            text_inputs: [text_batch_size, text_seq_len]
            labels: [text_batch_size]

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
            # token_type_ids=text_inputs["token_type_ids"].to(self.device),
        )
        text_pooler_out = text_encoded.last_hidden_state[:, 0, :]
        text_embedded = self.text_proj(text_pooler_out)

        # table as query list, text as paired
        loss = self.criterion(anchors=table_embedded, positives=text_embedded, labels=labels)
        return loss


def train(model: TableCL, optimizer, dataloader, args):
    tb = SummaryWriter(args.tensorboard_dir)
    model.train()
    total_step = args.start_total_step
    lg.info(f"start total step: {total_step}")

    for epoch in range(1, args.epochs + 1):  # start from 1
        report_loss = 0.
        for step, batch in enumerate(dataloader):
            loss = model(*batch)
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
    # continue
    parser.add_argument("--load_checkpoint", type=str, default=None, help="load checkpoint for both model and optimizer")
    parser.add_argument("--start_total_step", type=int, default=1, help="start from total_step")

    # data
    parser.add_argument("--max_title_length", type=int, default=128)  # todo? table max len?
    parser.add_argument("--aug", nargs="*", type=str, choices=["w2v", "syno", "trans"], help="augment title for more positive samples")
    parser.add_argument("--aug_dir", type=str, default="output/data/aug")

    # huggingface
    parser.add_argument("--table_model", type=str, default="google/tapas-small")
    parser.add_argument("--text_model", type=str, default="distilbert-base-uncased")

    # model
    parser.add_argument("--uni_dim", type=int, default=512, help="projection dim for both modality")
    parser.add_argument("--text_hidden_dim", type=int, default=768, help="bert output dim")
    parser.add_argument("--table_hidden_dim", type=int, default=512, help="tapas output dim")

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
    num_workers = 0 if args.debug else 8
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=args.shuffle,
                                                   collate_fn=collate_fn,
                                                   num_workers=num_workers)

    model = TableCL(args)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint + ".pth"))
        optimizer.load_state_dict(torch.load(args.load_checkpoint + ".opt"))
        lg.info(f"[LOAD] load model and optimizer from {args.load_checkpoint}")
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
