#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: collect_results.py
# --- Creation Date: 27-08-2020
# --- Last Modified: Sun 08 Nov 2020 16:33:25 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Collect results from a directory.
"""

import argparse
import os
import pdb
import glob
import numpy as np
import pickle
import pandas as pd
import re

from collections import OrderedDict
from metrics.metric_defaults import metric_defaults

# Metrics entries of interest.
# If not shown here, means all entries of the metrics are of interest.
moi = {'tpl': ['sum_dist'], 'fvm': ['eval_acc', 'act_dim']}

# Brief dict of names.
brief = {'beta_vae_modular': 'btv', 'factor_vae_modular': 'fcv',
         'factor_sindis_vae_modular': 'fcsv',
         'dip_vae_i_modular': 'dvi', 'dip_vae_ii_modular': 'dvii',
         'betatc_vae_modular': 'btcv', 'group_vae_modular': 'gpv',
         'l2_loss': 'l2', 'bernoulli_loss': 'bno',
         'Standard_prior_norelu_G': 'nreluG',
         'Standard_prior_G': 'stdG',
         'Group_prior_G': 'gpG', 'Group_prior_sim_G': 'gpsimG',
         'factorvae_dsprites_all': 'fvm',
         'factorvae_dsprites_all_vae': 'fvm', 'group_verify': 'gv',
         'factorvae_dsprites_all_hpc': 'fvm',
         'factorvae_dsprites_all_hpc_vae': 'fvm',
         'factorvae_shape3d_all_hpc_vae': 'fvm',
         'factorvae_dsprites_all_devcube2_vae': 'fvm',
         'factorvae_dsprites_all_devcube2': 'fvm',
         'factorvae_shape3d_all_hpc': 'fvm',
         'factorvae_shape3d_all_vae': 'fvm',
         'factorvae_shape3d_all': 'fvm',
         'tpl_nomap': 'tpl'}

def extend_exist_metrics_for_new_config(results):
    for key in results:
        results[key].append([])
    return results

def fill_configs_for_new_metric(results, new_metric):
    keys = list(results.keys())
    if len(keys) == 0:
        results[new_metric] = [[]]
    else:
        results[new_metric] = [[] * len(results[keys[0]])]
    return results

def get_mean(x):
    x = list(filter(None, x))
    return None if len(x) == 0 else np.mean(x)

def get_num(x):
    x = list(filter(None, x))
    return len(x)

def get_std(x):
    x = list(filter(None, x))
    return None if len(x) == 0 else np.std(x)

def get_mean_std(results):
    '''
    Calculate mean and std based on the raw data with different seeds.
    args: results: {'fvm.eval_acc': [[0.7, 0.8], [0.4, 0.8]],
                    'fvm.n_dim': [[5, 6], [3, 5]]}
    return: new_results: {'fvm.eval_acc.mean': [0.75, 0.6], ...}
    '''
    new_results = {}
    for k, v in results.items():
        k_mean, k_std = k+'.mean', k+'.std'
        v_mean = [get_mean(x) for x in v]
        v_std = [get_std(x) for x in v]
        new_results[k_mean] = v_mean
        new_results[k_std] = v_std
    new_results['num_samples'] = [get_num(x) for x in results[list(results.keys())[0]]]
    return new_results

def parse_config_v(config):
    config = config[1:-1]
    config = [x.strip().split('.') for x in config.split(',')]
    # config: [['run_desc'], ['run_func_kwargs', 'G_args', 'module_G_list'], ...]
    return config

def extract_v(data_dict, config_ls):
    if len(config_ls) == 1:
        if config_ls[0] == 'run_desc':
            return data_dict[config_ls[0]].split('-')[0]
        elif config_ls[0] == 'module_G_list':
            return data_dict[config_ls[0]][0].split('-')[0]
        else:
            if config_ls[0] in data_dict:
                return data_dict[config_ls[0]]
            else:
                return 0
    return str(extract_v(data_dict[config_ls[0]], config_ls[1:]))

def simplify_conf_name(name):
    return brief.get(name, name)

def get_config(dir_name, config_variables):
    with open(os.path.join(dir_name, 'submit_config.pkl'), 'rb') as f:
        s = f.read()
        data_dict = pickle.loads(s)
    config_vs = []
    for config_ls_i in config_variables:
        config_vs.append(simplify_conf_name(extract_v(data_dict, config_ls_i)))
    return '-'.join(config_vs)

def extract_this_results(dir_name, target_step):
    # this_results: {'fvm.eval_acc': 0.5, 'fvm.n_dim': 4, ...}
    # moi = {'tpl': ['sum_dist'], 'fvm': ['eval_acc', 'act_dim']}
    results = {}
    for metric in metric_defaults:
        met_fname = os.path.join(dir_name, 'metric-' + metric + '.txt')
        if os.path.exists(met_fname):
            with open(met_fname, 'r') as f:
                data = f.readlines()
            line_ls = re.split(' +', data[-1])
            if int(line_ls[0].split('-')[-1]) == target_step:
                cum_idx = 0
                met_brief = brief.get(metric, metric)
                for i, item in enumerate(line_ls):
                    if item.startswith(metric):
                        if met_brief in moi:
                            for sub in moi[met_brief]:
                                if sub in item:
                                    results[met_brief+'.'+sub] = float(line_ls[i+1])
                                    break
                        else:
                            results[met_brief+'.'+str(cum_idx)] = float(line_ls[i+1])
                            cum_idx += 1
    return results

def is_smaller(a, b):
    return a < b

def is_larger(a, b):
    return b < a

def get_max_metric_step(dir_name, metric, sub, compare_fn):
    met_fname = os.path.join(dir_name, 'metric-' + metric + '.txt')
    if compare_fn == is_smaller:
        v_optimal = float('inf')
    else:
        v_optimal = -float('inf')
    if os.path.exists(met_fname):
        with open(met_fname, 'r') as f:
            data = f.readlines()
        target_step = 0
        for line in data:
            line_ls = re.split(' +', line)
            metric_and_sub = '_'.join((metric, sub))
            for i, item in enumerate(line_ls):
                if item == metric_and_sub:
                    v_i = float(line_ls[i+1])
                    if compare_fn(v_i, v_optimal):
                        v_optimal = v_i
                        target_step = int(line_ls[0].split('-')[-1])
    return target_step

def main():
    parser = argparse.ArgumentParser(description='Collect results.')
    parser.add_argument('--in_dir', help='Parent directory of sub-result-dirs to collect results.',
                        type=str)
    parser.add_argument('--result_file', help='Results file.',
                        type=str, default='/mnt/hdd/repo_results/test.csv')
    parser.add_argument('--config_variables', help='Configs to extract from submit_config.pkl',
                        type=str, default=\
                        '[run_desc, run_func_kwargs.G_args.module_G_list, run_func_kwargs.G_loss_args.group_loss_type, run_func_kwargs.G_loss_args.hy_beta]')
    parser.add_argument('--target_step', help='Target step to extract.',
                        type=int, default=20000)
    parser.add_argument('--optimal_metric', help='The metric to get optimal target step. If is None, use --target_step.',
                        type=str, default=None)
    parser.add_argument('--optimal_sub', help='The metric sub value to get target step.',
                        type=str, default=None)
    parser.add_argument('--small_or_large', help='The metric small or large is good.',
                        type=str, default='large')
    args = parser.parse_args()

    args.config_variables = parse_config_v(args.config_variables)
    res_dirs = glob.glob(os.path.join(args.in_dir, '0*/'))
    res_dirs.sort()
    results = {}
    # results: {'fvm.eval_acc': [[0.7, 0.8], [0.4, 0.8]],
    #           'fvm.n_dim': [[5, 6], [3, 5]]}
    config_ls = [] # ['beta-1', 'beta-2']

    if args.small_or_large == 'small':
        compare_fn = is_smaller
    else:
        compare_fn = is_larger
    for dir_name in res_dirs:
        config = get_config(dir_name, args.config_variables)
        if args.optimal_metric:
            target_step = get_max_metric_step(dir_name, args.optimal_metric, args.optimal_sub, compare_fn)
        else:
            target_step = args.target_step
        print('target_step:', target_step)
        this_results = extract_this_results(dir_name, target_step)
        if this_results != {}:
            if config not in config_ls:
                config_ls.append(config)
                results = extend_exist_metrics_for_new_config(results)
            idx_config = config_ls.index(config)
        # this_results: {'fvm.eval_acc': 0.5, 'fvm.n_dim': 4, ...}
        for k, v in this_results.items():
            if k not in results.keys():
                results = fill_configs_for_new_metric(results, k)
            results[k][idx_config].append(v)

    for k, v in results.items():
        assert len(v) == len(config_ls)
    new_results = get_mean_std(results)
    new_results['_config'] = config_ls
    new_results = OrderedDict(sorted(new_results.items()))
    results_df = pd.DataFrame(new_results)
    print('results_df:', results_df)
    results_df.to_csv(args.result_file, na_rep='-',
                      index=False, float_format='%.3f')


if __name__ == "__main__":
    main()
