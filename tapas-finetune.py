import os
import argparse
from functools import partial

import torch
import torch.utils.data
from tqdm import tqdm
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer

from utils.metrics import SqaMetric
from dataloader import TableDataset, collate_fn


def train(model, train_dataloader, valid_dataloader, test_dataloader, tokenizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.to(device)
    model.train()

    total_step = 0
    for epoch in range(10):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        data_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=True)
        for idx, batch in data_iter:
            # get the inputs;
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            loss = outputs.loss
            # data_iter.set_postfix(loss=loss.item())
            print(f"[TRAIN] epoch: {epoch}, step: {idx}, loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            if total_step % 100 == 0:
                valid_loss = evaluate(model, valid_dataloader, tokenizer, args.valid_tsv)
                print(f"[VALID] epoch: {epoch}, step: {idx}, loss: {valid_loss}")

            total_step += 1


def evaluate(model, test_dataloader, tokenizer, tsv_path):
    print("Evaluating...", end=" ")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sqa_metric = SqaMetric(tsv_path)
    total_loss = 0.
    with torch.no_grad():
        for bid, batch in enumerate(test_dataloader):
            if bid % 10 == 0:
                print(bid, end=" ")
            if bid == len(test_dataloader) - 1:
                print()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            metadatas = batch["metadata"]
            origin_encodings = batch["origin_encoding"]

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            total_loss += outputs.loss.item()
            for metadata, origin_encoding, logit in zip(metadatas, origin_encodings, outputs.logits):
                logit = logit.unsqueeze(0).cpu().detach()
                predicted_answer_coordinates, = tokenizer.convert_logits_to_predictions(origin_encoding, logit)
                assert len(predicted_answer_coordinates) == 1
                ans_cord = predicted_answer_coordinates[0]
                ans_cord = set(ans_cord)
                sqa_metric.add_pred(metadata, ans_cord)
    sqa_metric.get_acc()
    model.train()
    return total_loss / len(test_dataloader)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default="data/SQA/table_csv")
    parser.add_argument("--train_tsv", type=str, default="data/SQA/random-split-1-train.tsv")
    parser.add_argument("--valid_tsv", type=str, default="data/SQA/random-split-1-dev.tsv")
    parser.add_argument("--test_tsv", type=str, default="data/SQA/test.tsv")

    return parser


def main():
    sqa_path = "./data/SQA"
    model_name = "google/tapas-base"
    # model_name = "google/tapas-base-finetuned-sqa"

    parser = get_parser()
    args = parser.parse_args()

    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)

    train_dataset = TableDataset(csv_dir=args.csv_dir,
                                 tsv_path=args.train_tsv,
                                 tokenizer=tokenizer)
    valid_dataset = TableDataset(csv_dir=args.csv_dir,
                                 tsv_path=args.valid_tsv,
                                 tokenizer=tokenizer,
                                 is_eval=True)
    test_dataset = TableDataset(csv_dir=args.csv_dir,
                                tsv_path=args.test_tsv,
                                tokenizer=tokenizer,
                                is_eval=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, is_eval=False))
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, collate_fn=partial(collate_fn, is_eval=True))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, is_eval=True))

    train(model, train_dataloader, valid_dataloader, test_dataloader, tokenizer, args)  # train


if __name__ == "__main__":
    main()
