import os
import logging
import argparse
from functools import partial

import torch
import torch.utils.data
from tqdm import tqdm
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer

from utils.metrics import SqaMetric
from utils.util import init_logging
from dataloader import TableDataset, collate_fn


lg = logging.getLogger()


def train(model, train_dataloader, valid_dataloader, test_dataloader, tokenizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.to(device)
    model.train()

    total_step = 0
    best_valid_ans_acc = 0.
    model_name = '?'
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        lg.info(f"Epoch: {epoch}")
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
            lg.info(f"[TRAIN] epoch: {epoch}, step: {idx} / {len(train_dataloader)}, loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            total_step += 1

        # evaluate each epoch
        valid_loss, valid_seq_acc, valid_ans_acc = evaluate(model, valid_dataloader, tokenizer, args.valid_tsv)
        lg.info(f"[VALID] epoch: {epoch}, step: {total_step}, loss: {valid_loss}, seq_acc: {valid_seq_acc}, ans_acc: {valid_ans_acc}")
        if valid_ans_acc > best_valid_ans_acc:
            best_valid_ans_acc = valid_ans_acc

            # test
            test_loss, test_seq_acc, test_ans_acc = evaluate(model, test_dataloader, tokenizer, args.test_tsv)
            lg.info(f"[TEST] epoch: {epoch}, step: {total_step}, loss: {test_loss}, seq_acc: {test_seq_acc}, ans_acc: {test_ans_acc}")

            # save
            model_name = f"{epoch}_{total_step}_{valid_ans_acc:.4f}_{test_ans_acc:.4f}"
            save_path = os.path.join(args.checkpoints, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)
            lg.info(f"[SAVE] {save_path}")
    # TODO collect best result
    lg.info(f"[BEST] {model_name}")


def evaluate(model, test_dataloader, tokenizer, tsv_path):
    lg.info(f"Start evaluating {tsv_path} ...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sqa_metric = SqaMetric(tsv_path)
    total_loss = 0.
    with torch.no_grad():
        for bid, batch in enumerate(test_dataloader):
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
    seq_acc, ans_acc = sqa_metric.get_acc()
    model.train()
    return total_loss / len(test_dataloader), seq_acc, ans_acc


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default="data/SQA/table_csv")
    parser.add_argument("--train_tsv", type=str, default="data/SQA/random-split-1-train.tsv")
    parser.add_argument("--valid_tsv", type=str, default="data/SQA/random-split-1-dev.tsv")
    parser.add_argument("--test_tsv", type=str, default="data/SQA/test.tsv")
    parser.add_argument("--output_dir", type=str, default="output/0508/0_demo")
    parser.add_argument("--shuffle", action="store_true", help="shuffle training data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--model_name", type=str, default="google/tapas-small")
    parser.add_argument("--pretrain_model", type=str, help="pretrain model containing tapas table encoder")

    parser.add_argument("--debug", action="store_true")
    return parser


def make_config(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"{args.output_dir} is created")
    args.checkpoints = os.path.join(args.output_dir, "checkpoints")
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
        print(f"{args.checkpoints} is created")
    args.log_path = os.path.join(args.output_dir, "train.log")
    init_logging(args.log_path, args.debug)


def get_dataloader(args, tokenizer):
    train_dataset = TableDataset(
        csv_dir=args.csv_dir,
        tsv_path=args.train_tsv,
        tokenizer=tokenizer
    )
    valid_dataset = TableDataset(
        csv_dir=args.csv_dir,
        tsv_path=args.valid_tsv,
        tokenizer=tokenizer,
        is_eval=True
    )
    test_dataset = TableDataset(
        csv_dir=args.csv_dir,
        tsv_path=args.test_tsv,
        tokenizer=tokenizer,
        is_eval=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=partial(collate_fn, is_eval=False)
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, is_eval=True)
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, is_eval=True)
    )
    return train_dataloader, valid_dataloader, test_dataloader


def init_model_from_pretrain(model, pretrain_model):
    pretrained_dict = torch.load(pretrain_model)
    # for param_name, param in pretrained_dict.items():
    #     print(param_name, '\t', param.shape)
    model_dict = model.state_dict()
    for param_name, param in model_dict.items():
        # print(param_name, '\t', param.shape)
        if param_name.startswith('tapas'):
            pretrained_name = param_name.replace('tapas.', 'table_encoder.', 1)
            lg.debug(f"{param_name} is copied from {pretrained_name}")
            model_dict[param_name] = pretrained_dict[pretrained_name]
    model.load_state_dict(model_dict)
    del pretrained_dict


def main():
    parser = get_parser()
    args = parser.parse_args()
    make_config(args)
    lg.info("=" * 50)
    lg.info(args)

    tokenizer = TapasTokenizer.from_pretrained(args.model_name)
    model = TapasForQuestionAnswering.from_pretrained(args.model_name)

    # load pretrained parameters
    if args.pretrain_model:
        init_model_from_pretrain(model, args.pretrain_model)
        lg.info(f"{args.pretrain_model} is loaded into tapas")

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args, tokenizer)
    train(model, train_dataloader, valid_dataloader, test_dataloader, tokenizer, args)  # train


if __name__ == "__main__":
    main()
