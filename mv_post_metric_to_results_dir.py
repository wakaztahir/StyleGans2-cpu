#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: mv_post_metric_to_results_dir.py
# --- Creation Date: 11-11-2020
# --- Last Modified: Mon 16 Nov 2020 23:35:26 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""

import argparse
import os
import pdb


def main():
    parser = argparse.ArgumentParser(description='Project description.')
    parser.add_argument('--results_dir',
                        help='Results directory.',
                        type=str,
                        default='/mnt/hdd/repo_results/test')
    parser.add_argument('--post_metric_dir',
                        help='Post calculated metric dir.',
                        type=str,
                        default='/mnt/hdd/Datasets/test_data')
    parser.add_argument('--new_metric_file',
                        help='The new-to-write metric file.',
                        type=str,
                        default='/mnt/hdd/test')
    args = parser.parse_args()
    post_metric_logfile = os.path.join(args.post_metric_dir, 'log.txt')
    with open(post_metric_logfile, 'r') as f:
        data = f.readlines()
    new_metric_line = data[-2]
    new_metric_filepath = os.path.join(args.results_dir, args.new_metric_file)
    # with open(new_metric_filepath, 'a') as f:
    with open(new_metric_filepath, 'w') as f:
        f.write(new_metric_line)


if __name__ == "__main__":
    main()
