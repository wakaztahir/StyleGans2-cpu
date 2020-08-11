#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: show_real_data.py
# --- Creation Date: 11-08-2020
# --- Last Modified: Tue 11 Aug 2020 14:51:08 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Show some real data.
"""

import argparse
import dnnlib
from training import dataset
from training import misc
from dnnlib import EasyDict


def show_real_data(data_dir, dataset, number):
    dataset_args = EasyDict(tfrecord_dir=dataset, max_label_size='full')
    training_set = dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir),
                                        verbose=True,
                                        **dataset_args)
    gw = 1
    gh = 1
    for i in range(number):
        reals, _ = training_set.get_minibatch_np(gw * gh)
        misc.save_image_grid(reals,
                             dnnlib.make_run_dir_path('reals%04d.png' % (i)),
                             drange=[0, 1],
                             grid_size=None)


def main():
    parser = argparse.ArgumentParser(
        description='Show real data',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result-dir',
                        help='Dir to save real imgs',
                        default='results',
                        metavar='DIR')
    parser.add_argument('--data-dir',
                        help='Dataset root directory',
                        required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--number',
                        help='Img number',
                        default=5,
                        metavar='NUM')
    args = parser.parse_args()

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    # sc.run_desc = kwargs.pop('command')

    dnnlib.submit_run(sc, 'show_real_data.show_real_data', **kwargs)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
