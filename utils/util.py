# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: util.py
@version: 1.0
@time: 2022/05/07 19:49:49
@contact: jinxy@pku.edu.cn

common tools
"""
import logging
import pathlib

import torch
import torch.backends.cudnn
import numpy as np
import random


def init_logging(root_log_path, debug=False, logger_name=None, sum_log_path=None):
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s',
                            datefmt='%m-%d %H:%M:%S')
    if logger_name:  # not root logger
        logger = logging.getLogger(logger_name)
    else:  # root logger
        logger = logging.getLogger()
    hdl = logging.FileHandler(root_log_path)
    hdl.setFormatter(fmt)
    logger.addHandler(hdl)
    logger.setLevel(logging.INFO)
    if debug:
        debug_hdl = logging.StreamHandler()
        debug_hdl.setFormatter(fmt)
        logger.addHandler(debug_hdl)
        logger.setLevel(logging.DEBUG)

    if sum_log_path:
        sum_logger = logging.getLogger('sum')
        sum_hdl = logging.FileHandler(sum_log_path)
        sum_hdl.setFormatter(fmt)
        sum_logger.addHandler(sum_hdl)
        sum_logger.setLevel(logging.INFO)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_config(args):
    setup_seed(args.seed)

    args.device = torch.device(args.device)
    args.output_dir = pathlib.Path(args.output_dir)
    args.checkpoint_dir = args.output_dir / "checkpoints"
    args.tensorboard_dir = args.output_dir / "tensorboard"
    args.log_path = args.output_dir / "train.log"
    args.eval_log_path = args.output_dir / "eval.log"
    if not args.output_dir.exists():
        args.output_dir.mkdir()
        print(f"create output dir: {args.output_dir}")
    if not args.checkpoint_dir.exists():
        args.checkpoint_dir.mkdir()
        print(f"create checkpoint dir: {args.checkpoint_dir}")
    if not args.tensorboard_dir.exists():
        args.tensorboard_dir.mkdir()
        print(f"create tensorboard dir: {args.tensorboard_dir}")
    print(f"log path: {args.log_path}")
