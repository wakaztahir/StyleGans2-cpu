#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_training_infernet.py
# --- Creation Date: 26-05-2020
# --- Last Modified: Wed 27 May 2020 23:10:57 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Run training file for mapping a generator to latent codes (inference net).
Code borrowed from run_training.py from NVIDIA.
"""

import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults
from training.vc_modular_networks2 import split_module_names, LATENT_MODULES

#----------------------------------------------------------------------------


def run(result_dir, num_gpus, total_kimg,
        mirror_augment, metrics, resume_pkl,
        G_pkl, I_fmap_base=8, fmap_decay=0.15,
        n_samples_per=10, module_list=None,
        latent_type='uniform', batch_size=32, batch_per_gpu=16,
        random_seed=1000, fmap_min=16, fmap_max=512,
        dlatent_size=10, I_nf_scale=4, arch='resnet'):
    print('module_list:', module_list)
    train = EasyDict(run_func_name='training.training_loop_infernet.training_loop_infernet'
                     )  # Options for training loop.

    module_list = _str_to_list(module_list)
    I = EasyDict(func_name='training.vc_networks2.infer_modular',
                 dlatent_size=dlatent_size, fmap_min=fmap_min,
                 fmap_max=fmap_max, module_list=module_list,
                 I_nf_scale=I_nf_scale)
    desc = 'inference_net'

    I_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)  # Options for discriminator optimizer.
    loss = EasyDict(func_name='training.loss_inference.I_loss',
        latent_type=latent_type, dlatent_size=dlatent_size)  # Options for generator loss.

    sched = EasyDict()  # Options for TrainingSchedule.
    sc = dnnlib.SubmitConfig()  # Options for dnnlib.submit_run().
    # tf_config = {'rnd.np_random_seed': 1000}  # Options for tflib.init_tf().
    tf_config = {'rnd.np_random_seed': random_seed}  # Options for tflib.init_tf().

    train.total_kimg = total_kimg
    sched.lrate = 0.002
    sched.tick_kimg = 1
    sched.minibatch_size = batch_size
    sched.minibatch_gpu = batch_per_gpu
    metrics = [metric_defaults[x] for x in metrics]

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    # Configs A-E: Shrink networks to match original StyleGAN.
    I.fmap_base = 2 << I_fmap_base

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(I_args=I, I_opt_args=I_opt,
                  loss_args=loss)
    kwargs.update(sched_args=sched, metric_arg_list=metrics,
                  tf_config=tf_config, resume_pkl=resume_pkl, G_pkl=G_pkl,
                  n_samples_per=n_samples_per)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)


#----------------------------------------------------------------------------


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _str_to_list(v):
    v_values = v.strip()[1:-1]
    module_list = [x.strip() for x in v_values.split(',')]
    return module_list

def _str_to_list_of_int(v):
    v_values = v.strip()[1:-1]
    step_list = [int(x.strip()) for x in v_values.split(',')]
    return step_list


def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train VCGAN and INFOGAN.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--result-dir',
        help='Root directory for run results (default: %(default)s)',
        default='results',
        metavar='DIR')
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)',
                        default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg',
        help='Training length in thousands of images (default: %(default)s)',
        metavar='KIMG', default=25000, type=int)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)',
                        default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument(
        '--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)',
        default='None', type=_parse_comma_sep)
    parser.add_argument('--resume_pkl', help='Continue training using pretrained pkl.',
                        default=None, metavar='RESUME_PKL', type=str)
    parser.add_argument('--G_pkl', help='G to load.',
                        default=None, metavar='G_PKL', type=str)
    parser.add_argument('--n_samples_per', help='Number of samples for each line in traversal (default: %(default)s)',
        metavar='N_SHOWN_SAMPLES_PER_LINE', default=10, type=int)
    parser.add_argument('--module_list', help='Module list for modular network.',
                        default=None, metavar='MODULE_LIST', type=str)
    parser.add_argument('--batch_size', help='N batch.',
                        metavar='N_BATCH', default=32, type=int)
    parser.add_argument('--batch_per_gpu', help='N batch per gpu.',
                        metavar='N_BATCH_PER_GPU', default=16, type=int)
    parser.add_argument('--latent_type', help='What type of latent priori to use.',
                        metavar='LATENT_TYPE', default='uniform', choices=['uniform', 'normal', 'trunc_normal'], type=str)
    parser.add_argument('--fmap_decay', help='fmap decay for network building.',
                        metavar='FMAP_DECAY', default=0.15, type=float)
    parser.add_argument('--I_fmap_base', help='Fmap base for I.',
                        metavar='I_FMAP_BASE', default=8, type=int)
    parser.add_argument('--random_seed', help='TF random seed.',
                        metavar='RANDOM_SEED', default=9, type=int)
    parser.add_argument('--fmap_min', help='FMAP min.',
                        metavar='FMAP_MIN', default=16, type=int)
    parser.add_argument('--fmap_max', help='FMAP max.',
                        metavar='FMAP_MAX', default=512, type=int)
    parser.add_argument('--I_nf_scale', help='N feature map scale for I.',
                        metavar='I_NF_SCALE', default=4, type=int)
    parser.add_argument('--dlatent_size', help='Latent size. Used for vc2_gan_style2.',
                        metavar='DLATENT_SIZE', default=24, type=int)
    parser.add_argument('--arch', help='Architecture for vc2_gan_style2.',
                        metavar='ARCH', default='resnet', type=str)

    args = parser.parse_args()

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
