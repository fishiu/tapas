# -*- coding: utf-8 -*-

""" 
@author: Jin.Fish
@file: pipeline.py
@version: 1.0
@time: 2022/05/10 17:24:55
@contact: jinxy@pku.edu.cn

pipeline for data preprocessing
"""
import pathlib
import time

import bs4
import pandas as pd
from bs4 import BeautifulSoup as bs
from pylatexenc.latex2text import LatexNodes2Text


class TableParser:
    def __init__(self, debug=False):
        self.latex_converter = LatexNodes2Text()
        self.debug = debug

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
    def valid_table(table: bs4.element.Tag):
        # test and found no performance issue
        figure_node = table.find_parent('figure')
        if not figure_node:  # no parent figure node
            # # substitute caption node
            # figure_node = table.find_next_siblings('p', {'class': 'ltx_p'})
            # if not figure_node:  # figure_node == []
            #     return "no fig node and no p node"
            return "no fig node"

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
        print(f"\n\nparsed {len(tables)} tables from {html_file}")

        # loop tables
        for i, table in enumerate(tables):
            valid_state = self.valid_table(table)
            if valid_state:
                print(f"[WARNING] {html_file} table {i} is not valid, because {valid_state}")
                continue

            self.scrape_math(table)  # clean table content
            caption = self.fetch_caption(table)  # get caption
            df = pd.read_html(str(table))[0]  # todo missing header
            csv_path = csv_dir / f"{html_file.name}_{i}.csv"
            df.to_csv(csv_path, index=False, header=False)
            print(f"convert table {i} to {csv_path}")
            table_res_list.append({'caption': caption, 'csv_path': csv_path})  # save metadata
        print(f"convert {html_file} to {csv_dir} in {time.time() - start_time}s")
        return table_res_list

    def fetch_caption(self, table: bs4.element.Tag):
        """find parent node figcaption and get content text"""
        figure_node = table.find_parent('figure', {'class': 'ltx_table'})
        if figure_node:
            self.scrape_math(figure_node.figcaption)  # clean caption
            caption = figure_node.figcaption.get_text()
        else:  # handle case 1712.04621v1
            raise Exception(f"this should not happen")
            figure_node = table.find_next_siblings('p', {'class': 'ltx_p'})
            if len(figure_node) > 1:
                print("[WARNING] more than one figcaption")
            figure_node = figure_node[0]
            self.scrape_math(figure_node)
            caption = figure_node.get_text()
        return caption

    def pipeline(self, html_dir, csv_dir):
        html_dir = pathlib.Path(html_dir)
        csv_dir = pathlib.Path(csv_dir)
        log_path = csv_dir / "success.txt"
        if not log_path.exists():
            log_path.touch()
        with log_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            success_list = [line.strip() for line in lines]
        html_file_list = list(html_dir.glob("*"))
        print(f"get {len(html_file_list)} html files")

        res_list = list()
        for html_file in html_file_list:
            if html_file.name in success_list:
                print(f"[INFO] {html_file} already converted")
                continue
            csv_sub_dir = csv_dir / html_file.name
            if not csv_sub_dir.exists():
                csv_sub_dir.mkdir()
            res = self.html2csv(html_file, csv_sub_dir)
            res_list.extend(res)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{html_file.name}\n")

        df = pd.DataFrame(res_list)
        df.to_csv(csv_dir / "meta.csv", index=False)


def main():
    start = time.time()
    parser = TableParser(debug=True)
    parser.pipeline("../ar5iv_data", "../csv")
    print(f"done in {time.time() - start}s")

    # html_path = pathlib.Path("../ar5iv_data/1712.04910v1")
    # csv_path = pathlib.Path("../csv") / html_path.name
    # parser.html2csv(html_path, csv_path)


if __name__ == "__main__":
    main()
