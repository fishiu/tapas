# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: pipeline.py
@version: 1.0
@time: 2022/05/10 17:24:55
@contact: jinxy@pku.edu.cn

pipeline for data preprocessing
"""
import os
import pathlib
import re
import time
import logging
import argparse
import collections
import multiprocessing

import bs4
import pandas as pd
from bs4 import BeautifulSoup as bs
from pylatexenc.latex2text import LatexNodes2Text
from utils.util import init_logging


class TableParser:
    def __init__(self, task_id, debug=False):
        self.latex_converter = LatexNodes2Text()
        self.debug = debug
        self.error_stat = list()
        self.lg = logging.getLogger(f"TASK{task_id}")

    def parse_alttext(self, alttext: str):
        """convert alttext to plain ascii"""
        plain_text = self.latex_converter.latex_to_text(alttext)
        return plain_text

    def scrape_math(self, table_node: bs4.element.Tag):
        """
        scrape math from table node
        """
        for math_node in table_node.find_all('math'):
            math_alttext = math_node.get('alttext')
            math_alttext = self.parse_alttext(math_alttext)
            math_node.replace_with(math_alttext)

    @staticmethod
    def remove_thead(table: bs4.element.Tag):
        body = table.find("tbody")
        if body is None:
            return
        table_heads = table.find_all("thead")
        if len(table_heads) > 0:  # thead exist
            head = table_heads[0].extract()
            head_rows = head.find_all("tr")
            for row in head_rows[::-1]:
                row = row.extract()
                for th in row.find_all("th"):
                    th.name = "td"
                body.insert(0, row)
        return table

    @staticmethod
    def valid_table(table: bs4.element.Tag):
        # test and found no performance issue
        figure_node = table.find_parent('figure')
        if not figure_node:  # no parent figure node
            # # substitute caption node
            # figure_node = table.find_next_siblings('p', {'class': 'ltx_p'})
            # if not figure_node:  # figure_node == []
            #     return "no fig node and no p node"
            return "no fig node"

        if 'class' not in figure_node.attrs:
            return "no class"

        if 'biography' in figure_node['class'] or 'ltx_figure' in figure_node['class']:
            return "other type node"

        caption = figure_node.find('figcaption', recursive=False)
        if not caption:
            return "no caption"

    def html2csv(self, html_file: pathlib.Path, csv_dir: pathlib.Path):
        """
        convert html file to csv file
        """
        table_res_list = list()
        start_time = time.time()

        # parse table
        with html_file.open("r", encoding="utf-8") as f:
            soup = bs(f, "html.parser")
        tables = soup.find_all("table", {'class': 'ltx_tabular'})
        self.lg.info(f"----------------------- parsed {len(tables)} tables from {html_file}")

        # loop tables
        for i, table in enumerate(tables):
            try:
                valid_state = self.valid_table(table)
            except Exception as e:
                self.error_stat.append(f"validation error")
                self.lg.error(f"{html_file} table {i} validation error: {e}")
                continue
            if valid_state:
                self.error_stat.append(valid_state)
                self.lg.warning(f"{html_file} table {i} is not valid, because {valid_state}")
                continue

            try:
                self.scrape_math(table)  # clean table content
            except Exception as e:
                self.error_stat.append(f"scrape table math error")
                self.lg.error(f"{html_file} table {i} scrape math error: {e}, skip")
                continue

            try:
                caption = self.fetch_caption(table)  # get caption
            except Exception as e:
                self.error_stat.append("caption unknown error")
                self.lg.error(f"{html_file} table {i} is not valid, because {e}, skip")
                continue

            if not caption:
                self.error_stat.append("no caption")
                self.lg.warning(f"{html_file} table {i} has no caption, skip")
                continue

            table = self.remove_thead(table)  # remove thead (fixed missing header)
            if not table:
                self.error_stat.append("no tbody")
                self.lg.error(f"{html_file} table {i} has no tbody, skip")
                continue

            try:
                df = pd.read_html(str(table))[0]
            except Exception as e:
                self.error_stat.append("pandas error")
                self.lg.error(f"{html_file} table {i} pandas error, because {e}, skip")
                continue

            csv_path = csv_dir / f"{html_file.name}_{i}.csv"
            df.to_csv(csv_path, index=False, header=False)
            self.lg.info(f"convert table {i} to {csv_path}")
            table_res_list.append({'caption': caption, 'csv_path': csv_path})  # save metadata
        self.lg.info(f"finish {html_file} in {time.time() - start_time}s")
        return table_res_list

    def fetch_caption(self, table: bs4.element.Tag):
        """find parent node figcaption and get content text"""
        figure_node = table.find_parent('figure', {'class': 'ltx_table'})
        if figure_node:
            self.scrape_math(figure_node.figcaption)  # clean caption
            caption = figure_node.figcaption.get_text()
            return caption
        else:
            return False

    def pipeline(self, html_dir: pathlib.Path, csv_dir: pathlib.Path):
        record_path = csv_dir / "success.txt"
        if not record_path.exists():
            record_path.touch()

        success_list = []
        # with record_path.open("r", encoding="utf-8") as f:
        #     lines = f.readlines()
        #     success_list = [line.strip() for line in lines]

        html_file_list = list(html_dir.glob("*"))
        self.lg.info(f"get {len(html_file_list)} html files")

        res_list = list()
        for idx, html_file in enumerate(html_file_list):
            if html_file.name in success_list:
                self.lg.info(f"[INFO] {html_file} already converted")
                continue
            csv_sub_dir = csv_dir / html_file.name
            if not csv_sub_dir.exists():
                csv_sub_dir.mkdir()
            res = self.html2csv(html_file, csv_sub_dir)
            res_list.extend(res)
            with record_path.open("a", encoding="utf-8") as f:
                f.write(f"{html_file.name}\n")
            self.lg.info(f"convert {len(res_list)} csv files, {idx}/{len(html_file_list)} html files done")

        df = pd.DataFrame(res_list)
        df.to_csv(csv_dir / "meta.csv", index=False)

    def report_error(self):
        counter = collections.Counter(self.error_stat)
        self.lg.info(counter.most_common())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html_dir", type=str, help="html dir", required=True)
    parser.add_argument("--csv_dir", type=str, help="csv dir", required=True)
    parser.add_argument("--proc_num", type=int, help="process num", default=1)
    parser.add_argument("--task_id", nargs="+", type=int, help="task id", required=True)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


def task(task_id: int, html_dir: pathlib.Path, csv_dir: pathlib.Path, debug: bool):
    if not csv_dir.exists():
        csv_dir.mkdir()
        print(f"[TASK{task_id}] create csv dir {csv_dir}")
    log_path = csv_dir / "parse.log"
    init_logging(root_log_path=log_path.absolute(), debug=debug, logger_name=f"TASK{task_id}")
    lg = logging.getLogger(f"TASK{task_id}")
    lg.info(f"start task {task_id}, html_dir: {html_dir}, csv_dir: {csv_dir}")
    start = time.time()
    parser = TableParser(task_id=task_id, debug=debug)
    parser.pipeline(html_dir, csv_dir)
    lg.info(f"done in {time.time() - start}s")
    parser.report_error()


def multi_run():
    args = get_args()
    print(args)
    html_dir = pathlib.Path(args.html_dir)
    csv_dir = pathlib.Path(args.csv_dir)
    p = multiprocessing.Pool(args.proc_num)
    for i in args.task_id:
        html_task_dir = html_dir / str(i)
        csv_task_dir = csv_dir / str(i)
        p.apply_async(task, args=(i, html_task_dir, csv_task_dir, args.debug))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


def convert_caption(caption):
    caption = re.sub(r"\s+", " ", caption)
    if res := re.match(r"^\s?\([a-zA-Z0-9]{1,3}\)\s?\:?\s?(.*)", caption):
        return res.group(1)
    if res := re.match(r"^\s?(?:TABLE|table|Table)\s*.{0,4}[\:\.]\s*(.*)", caption):
        return res.group(1)
    print(caption)
    return caption


def convert():
    csv_dir = pathlib.Path("../data/ar5iv_csv/")
    for d in csv_dir.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.csv"
        df = pd.read_csv(meta_path)
        df["title"] = df["caption"].apply(convert_caption)
        new_meta_path = d / "meta_title.csv"
        df.to_csv(new_meta_path, index=False)
        print(f"convert {meta_path} to {new_meta_path}")


if __name__ == "__main__":
    # multi_run()
    convert()
