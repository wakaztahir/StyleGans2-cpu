#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: collect_results.py
# --- Creation Date: 27-08-2020
# --- Last Modified: Fri 13 Nov 2020 16:09:20 AEDT
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
# moi = {'tpl': ['sum_dist'], 'fvm': ['eval_acc', 'act_dim'], 'mig': ['discrete_mig']}
moi = {'tpl': ['sum_dist'], 'fvm': ['eval_acc', 'act_dim'], 'tpl_large': ['sum_dist']}

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
         'tpl_nomap': 'tpl', 'mig_dsprites_all': 'mig',
         'np_random_seed': 'seed'
         }

def extend_exist_metrics_for_new_config(results):
    for key in results:
        results[key].append([])
    return results

def fill_configs_for_new_metric(results, new_metric):
    keys = list(results.keys())
    if len(keys) == 0:
        results[new_metric] = [[]]
    else:
        print('len(results[keys[0]]):', len(results[keys[0]]))
        results[new_metric] = [[] for i in range(len(results[keys[0]]))]
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

def _list_to_str(v):
    v2 = [str(x) for x in v]
    return ','.join(v2)

def get_mean_std(results):
    '''
    Calculate mean and std based on the raw data with different seeds.
    args: results: {'fvm.eval_acc': [[0.7, 0.8], [0.4, 0.8]],
                    'fvm.n_dim': [[5, 6], [3, 5]]}
    return: new_results: {'fvm.eval_acc.mean': [0.75, 0.6], ...}
    '''
    new_results = {}
    for k, v in results.items():
        if k == 'dir_id' or k == 'seed':
            new_results[k] = map(lambda x: str(x), v)
        else:
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
    if config_ls[0] == 'rnd' and config_ls[1] == 'np_random_seed' and len(config_ls) == 2:
        k = '.'.join(config_ls)
        if k in data_dict:
            return data_dict[k]
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
    seed = None
    for config_ls_i in config_variables:
        v = simplify_conf_name(extract_v(data_dict, config_ls_i))
        if config_ls_i[-1] == 'np_random_seed':
            seed = v
        else:
            config_vs.append(v)
    return '-'.join(config_vs), seed

def extract_this_results(dir_name, target_step):
    # this_results: {'fvm.eval_acc': 0.5, 'fvm.n_dim': 4, ...}
    # moi = {'tpl': ['sum_dist'], 'fvm': ['eval_acc', 'act_dim']}
    results = {}
    for metric in metric_defaults:
        met_brief = brief.get(metric, metric)
        if not met_brief in moi:
            continue
        met_fname = os.path.join(dir_name, 'metric-' + metric + '.txt')
        if os.path.exists(met_fname):
            with open(met_fname, 'r') as f:
                data = f.readlines()
            for line in data:
                line_ls = re.split(' +', line)
                if int(line_ls[0].split('-')[-1]) == target_step:
                    cum_idx = 0
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

def get_max_metric_step(dir_name, metric, sub, old_target_step, compare_fn):
    met_fname = os.path.join(dir_name, 'metric-' + metric + '.txt')
    if compare_fn == is_smaller:
        v_optimal = float('inf')
    else:
        v_optimal = -float('inf')
    # print('met_fname:', met_fname)
    target_step = 0
    if os.path.exists(met_fname):
        with open(met_fname, 'r') as f:
            data = f.readlines()
        found_target_step = False
        for line in data[-len(data)//3:]:
            line_ls = re.split(' +', line)
            metric_and_sub = '_'.join((metric, sub))
            step_i = int(line_ls[0].split('-')[-1])
            if step_i == old_target_step:
                found_target_step = True
            for i, item in enumerate(line_ls):
                if item == metric_and_sub:
                    v_i = float(line_ls[i+1])
                    if compare_fn(v_i, v_optimal):
                        v_optimal = v_i
                        target_step = int(line_ls[0].split('-')[-1])
        if not found_target_step:
            target_step = old_target_step
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
    if args.optimal_metric is not None:
        args.result_file = args.result_file[:-4]+'-'+str(args.target_step)+'-'+args.optimal_metric+'-'+'.csv'
    else:
        args.result_file = args.result_file[:-4]+'-'+str(args.target_step)+'-'+'.csv'
    res_dirs = glob.glob(os.path.join(args.in_dir, '0*/'))
    # print('res_dirs:', res_dirs)
    res_dirs.sort()
    results = {}
    # results: {'fvm.eval_acc': [[0.7, 0.8], [0.4, 0.8]],
    #           'fvm.n_dim': [[5, 6], [3, 5]]}
    config_ls = [] # ['beta-1', 'beta-2']
    raw_results = []

    if args.small_or_large == 'small':
        compare_fn = is_smaller
    else:
        compare_fn = is_larger
    for dir_name in res_dirs:
        config, seed = get_config(dir_name, args.config_variables)
        dir_id = int(os.path.basename(dir_name[:-1]).split('-')[0]) # remove last '/'
        if args.optimal_metric:
            target_step = get_max_metric_step(dir_name, args.optimal_metric,
                                              args.optimal_sub, args.target_step, compare_fn)
        else:
            target_step = args.target_step
        # print('target_step:', target_step)
        this_results = extract_this_results(dir_name, target_step)
        if this_results != {}:
            this_results['dir_id'] = dir_id
            this_results['seed'] = seed
            if config not in config_ls:
                config_ls.append(config)
                results = extend_exist_metrics_for_new_config(results)
            idx_config = config_ls.index(config)
        print('this_results:', this_results)
        # this_results: {'fvm.eval_acc': 0.5, 'fvm.n_dim': 4, ...}
        for k, v in this_results.items():
            if k not in results.keys():
                results = fill_configs_for_new_metric(results, k)
            print('idx_config:', idx_config)
            results[k][idx_config].append(v)
        raw_results.append(this_results)

    # raw_results_file = args.result_file[:-4] + '.txt'
    # # raw_results = [str(x) for x in raw_results]
    # raw_results = map(lambda x: str(x)+'\n', raw_results)
    # with open(raw_results_file, 'w') as f:
        # f.writelines(raw_results)
    for k, v in results.items():
        assert len(v) == len(config_ls)

    new_results = get_mean_std(results)
    save_results_to_csv(new_results, config_ls, args, '')
    save_results_to_csv(results, config_ls, args, '_raw')

def save_results_to_csv(results, config_ls, args, sufix):
    results['_config'] = config_ls
    results = OrderedDict(sorted(results.items()))
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.result_file[:-4]+sufix+'.csv', na_rep='-',
                      index=False, float_format='%.3f')


if __name__ == "__main__":
    main()
