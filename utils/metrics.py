# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: metrics.py
@version: 1.0
@time: 2022/05/07 21:21:52
@contact: jinxy@pku.edu.cn

calculating metrics
"""

import csv


class SqaMetric:
    def __init__(self, gold_path):
        self.dt_gold = self.read_tsv(gold_path)
        self.dt_pred = {}

    @staticmethod
    def read_tsv(fn_tsv):
        dt = {}
        for row in csv.DictReader(open(fn_tsv, 'r'), delimiter='\t'):
            sid = row['id'] + '\t' + row['annotator']  # sequence id
            pos = int(row['position'])  # position
            ans_cord = eval(row['answer_coordinates'])  # list
            ans_cord = {eval(i) for i in ans_cord}  # set of tuple
            if sid not in dt:
                dt[sid] = {}
            dt[sid][pos] = ans_cord
        return dt

    def add_pred(self, metadata: dict, ans_cord: set):
        idx, annotator, pos = metadata['id'], metadata['annotator'], int(metadata['position'])
        sid = idx + '\t' + str(annotator)  # sequence id
        if sid not in self.dt_pred:
            self.dt_pred[sid] = {}
        self.dt_pred[sid][pos] = ans_cord

    def get_acc(self):
        # Calculate both sequence-level accuracy and question-level accuracy
        seq_cnt = seq_cor = 0
        ans_cnt = ans_cor = 0
        for sid, qa in self.dt_gold.items():
            seq_cnt += 1
            ans_cnt += len(qa)

            if sid not in self.dt_pred:
                continue  # sequence does not exist in the prediction

            pred_qa = self.dt_pred[sid]
            all_q_correct = True
            for q, a in qa.items():
                if q in pred_qa and a == pred_qa[q]:
                    ans_cor += 1  # correctly answered question
                else:
                    all_q_correct = False
            if all_q_correct: seq_cor += 1

        # print("Sequence Accuracy = %0.2f%% (%d/%d)" % (100.0 * seq_cor / seq_cnt, seq_cor, seq_cnt))
        # print("Answer Accuracy =   %0.2f%% (%d/%d)" % (100.0 * ans_cor / ans_cnt, ans_cor, ans_cnt))
        # return [seq_cor, seq_cnt, ans_cor, ans_cnt]

        return ("Sequence Accuracy = %0.2f%% (%d/%d)" % (100.0 * seq_cor / seq_cnt, seq_cor, seq_cnt),
                "Answer Accuracy =   %0.2f%% (%d/%d)" % (100.0 * ans_cor / ans_cnt, ans_cor, ans_cnt))
