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


def init_logging(root_log_path, debug=False):
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s',
                            datefmt='%m-%d %H:%M:%S')
    root_logger = logging.getLogger()
    root_hdl = logging.FileHandler(root_log_path)
    root_hdl.setFormatter(fmt)
    root_logger.addHandler(root_hdl)
    root_logger.setLevel(logging.INFO)
    if debug:
        debug_hdl = logging.StreamHandler()
        debug_hdl.setFormatter(fmt)
        root_logger.addHandler(debug_hdl)
        root_logger.setLevel(logging.DEBUG)

    # eval_logger = logging.getLogger('evaluation')
    # eval_hdl = logging.FileHandler(eval_log_path)
    # eval_hdl.setFormatter(fmt)
    # eval_logger.addHandler(eval_hdl)
    # eval_logger.setLevel(logging.INFO)
