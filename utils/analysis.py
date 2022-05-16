# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: analysis.py
@version: 1.0
@time: 2022/05/16 02:24:38
@contact: jinxy@pku.edu.cn

analysis experiment result
"""


import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    log = pathlib.Path('../output/finetune/3_small/train.log')
    with log.open() as f:
        lines = f.readlines()
    best_acc = 0.
    best_line = ''
    for line in lines:
        line = line.strip()
        if "[VALID]" in line:
            print(line)
            ans_acc = eval(line.split(' ')[-1])
            print(ans_acc)
            if ans_acc > best_acc:
                best_acc = ans_acc
                best_line = line
    print('------------------------------------------------------')
    print(best_acc)
    print(best_line)


if __name__ == "__main__":
    main()
