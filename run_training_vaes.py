#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_training_vaes.py
# --- Creation Date: 13-08-2020
# --- Last Modified: Sat 15 Aug 2020 19:08:15 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Run training file Variational Autoencoders.
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


def run(dataset, data_dir, result_dir, num_gpus, total_kimg,
        mirror_augment, metrics, resume_pkl,
        model_type='vc_gan2', latent_type='uniform', batch_size=32,
        batch_per_gpu=16, random_seed=1000,
        G_fmap_base=8, module_G_list=None, G_nf_scale=4,
        E_fmap_base=8, module_E_list=None, E_nf_scale=4,
        D_fmap_base=9, module_D_list=None, D_nf_scale=4,
        fmap_decay=0.15, fmap_min=16, fmap_max=512,
        n_samples_per=10, arch='resnet', topk_dims_to_show=20,
        hy_beta=1, hy_gamma=0, G_lrate_base=0.002, D_lrate_base=None):
    train = EasyDict(
        run_func_name='training.training_loop_vae.training_loop_vae'
    )  # Options for training loop.

    if not (module_G_list is None):
        module_G_list = _str_to_list(module_G_list)
        key_G_ls, size_G_ls, count_dlatent_G_size = split_module_names(
            module_G_list)
    if not (module_E_list is None):
        module_E_list = _str_to_list(module_E_list)
        key_E_ls, size_E_ls, count_dlatent_E_size = split_module_names(
            module_E_list)
    if not (module_D_list is None):
        module_D_list = _str_to_list(module_D_list)
        key_D_ls, size_D_ls, count_dlatent_D_size = split_module_names(
            module_D_list)

    D = D_opt = D_loss = None
    if model_type == 'beta_vae':  # Beta-VAE
        E = EasyDict(func_name='training.vae_networks.E_main_modular',
                     fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay,
                     latent_size=count_dlatent_E_size,
                     module_E_list=module_E_list,
                     nf_scale=E_nf_scale,
                     fmap_base=2 << E_fmap_base)  # Options for encoder network.
        G = EasyDict(func_name='training.vae_networks.G_main_modular',
                     fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay,
                     latent_size=count_dlatent_G_size,
                     module_G_list=module_G_list,
                     nf_scale=G_nf_scale,
                     fmap_base=2 << G_fmap_base)  # Options for generator network.
        G_opt = EasyDict(beta1=0.0, beta2=0.99,
                         epsilon=1e-8)  # Options for generator optimizer.
        desc = 'beta_vae_net_modular'
    elif model_type == 'factor_vae':  # Factor-VAE
        E = EasyDict(func_name='training.vae_networks.E_main_modular',
                     fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay,
                     latent_size=count_dlatent_E_size,
                     module_E_list=module_E_list,
                     nf_scale=E_nf_scale,
                     fmap_base=2 << E_fmap_base)  # Options for encoder network.
        G = EasyDict(func_name='training.vae_networks.G_main_modular',
                     fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay,
                     latent_size=count_dlatent_G_size,
                     module_G_list=module_G_list,
                     nf_scale=G_nf_scale,
                     fmap_base=2 << G_fmap_base)  # Options for generator network.
        D = EasyDict(func_name='training.vae_networks.D_factor_vae_modular',
                     fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay,
                     latent_size=count_dlatent_D_size,
                     module_D_list=module_D_list,
                     nf_scale=D_nf_scale,
                     fmap_base=2 << D_fmap_base)  # Options for generator network.
        G_opt = EasyDict(beta1=0.9, beta2=0.999,
                         epsilon=1e-8)  # Options for generator optimizer.
        D_opt = EasyDict(beta1=0.5, beta2=0.9,
                         epsilon=1e-8)  # Options for discriminator optimizer.
        desc = 'factor_vae_net_modular'
    else:
        raise ValueError('Not supported model tyle: ' + model_type)

    if model_type == 'beta_vae':  # Beta-VAE
        G_loss = EasyDict(
            func_name='training.loss_vae.beta_vae',
            latent_type=latent_type, hy_beta=hy_beta)  # Options for generator loss.
    elif model_type == 'factor_vae':  # Factor-VAE
        G_loss = EasyDict(
            func_name='training.loss_vae.factor_vae_G',
            latent_type=latent_type, hy_gamma=hy_gamma)  # Options for generator loss.
        D_loss = EasyDict(
            func_name='training.loss_vae.factor_vae_D',
            latent_type=latent_type)  # Options for discriminator loss.

    sched = EasyDict()  # Options for TrainingSchedule.
    grid = EasyDict(size='1080p', layout='random')  # Options for setup_snapshot_image_grid().
    sc = dnnlib.SubmitConfig()  # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': random_seed, 'allow_soft_placement': True}  # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    sched.G_lrate_base = G_lrate_base
    sched.D_lrate_base = D_lrate_base
    sched.minibatch_size_base = batch_size
    sched.minibatch_gpu_base = batch_per_gpu
    metrics = [metric_defaults[x] for x in metrics]

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset, max_label_size='full')

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, E_args=E, D_args=D,
        G_opt_args=G_opt, D_opt_args=D_opt,
        G_loss_args=G_loss, D_loss_args=D_loss,
        traversal_grid=True)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid,
                  n_continuous=count_dlatent_G_size,
                  metric_arg_list=metrics, tf_config=tf_config, resume_pkl=resume_pkl,
                  n_samples_per=n_samples_per, topk_dims_to_show=topk_dims_to_show)
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
    module_G_list = [x.strip() for x in v_values.split(',')]
    return module_G_list


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
    parser = argparse.ArgumentParser(description='Train VAEs.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result-dir',
        help='Root directory for run results (default: %(default)s)',
        default='results', metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)',
                        default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg',
        help='Training length in thousands of images (default: %(default)s)',
        metavar='KIMG', default=25000, type=int)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)',
                        default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--metrics',
        help='Comma-separated list of metrics or "none" (default: %(default)s)',
        default='None', type=_parse_comma_sep)
    parser.add_argument('--model_type', help='Type of model to train', default='beta_vae',
                        type=str, metavar='MODEL_TYPE', choices=['beta_vae', 'factor_vae'])
    parser.add_argument('--resume_pkl', help='Continue training using pretrained pkl.',
                        default=None, metavar='RESUME_PKL', type=str)
    parser.add_argument('--n_samples_per',
        help='Number of samples for each line in traversal (default: %(default)s)',
        metavar='N_SHOWN_SAMPLES_PER_LINE', default=10, type=int)
    parser.add_argument('--batch_size', help='N batch.', metavar='N_BATCH',
                        default=32, type=int)
    parser.add_argument('--batch_per_gpu', help='N batch per gpu.', metavar='N_BATCH_PER_GPU',
                        default=16, type=int)
    parser.add_argument('--latent_type', help='What type of latent priori to use.', metavar='LATENT_TYPE',
                        default='normal', choices=['uniform', 'normal', 'trunc_normal'], type=str)
    parser.add_argument('--fmap_decay', help='fmap decay for network building.', metavar='FMAP_DECAY',
                        default=0.15, type=float)
    parser.add_argument('--G_fmap_base', help='Fmap base for G.', metavar='G_FMAP_BASE',
                        default=8, type=int)
    parser.add_argument('--E_fmap_base', help='Fmap base for E.', metavar='E_FMAP_BASE',
                        default=8, type=int)
    parser.add_argument('--D_fmap_base', help='Fmap base for D.', metavar='D_FMAP_BASE',
                        default=9, type=int)
    parser.add_argument('--random_seed', help='TF random seed.', metavar='RANDOM_SEED',
                        default=9, type=int)
    parser.add_argument('--module_G_list', help='Module list for G modular network.', default=None,
                        metavar='MODULE_G_LIST', type=str)
    parser.add_argument('--module_E_list', help='Module list for E modular network.', default=None,
                        metavar='MODULE_E_LIST', type=str)
    parser.add_argument('--module_D_list', help='Module list for D modular network.', default=None,
                        metavar='MODULE_D_LIST', type=str)
    parser.add_argument('--fmap_min', help='FMAP min.', metavar='FMAP_MIN',
                        default=16, type=int)
    parser.add_argument('--fmap_max', help='FMAP max.', metavar='FMAP_MAX',
                        default=512, type=int)
    parser.add_argument('--G_nf_scale', help='N feature map scale for G.', metavar='G_NF_SCALE',
                        default=4, type=int)
    parser.add_argument('--E_nf_scale', help='N feature map scale for E.', metavar='E_NF_SCALE',
                        default=4, type=int)
    parser.add_argument('--D_nf_scale', help='N feature map scale for D.', metavar='D_NF_SCALE',
                        default=4, type=int)
    parser.add_argument('--arch', help='Architecture for vc2_gan_style2_noI.', metavar='ARCH',
                        default='resnet', type=str)
    parser.add_argument('--topk_dims_to_show', help='Number of top disentant dimensions to show in a snapshot.',
        metavar='TOPK_DIMS_TO_SHOW', default=20, type=int)
    parser.add_argument('--hy_beta', help='Hyper-param for beta-vae.', metavar='HY_BETA',
                        default=1, type=float)
    parser.add_argument('--hy_gamma', help='Hyper-param for factor-vae.', metavar='HY_GAMMA',
                        default=0, type=float)
    parser.add_argument('--G_lrate_base', help='G learning rate.', metavar='G_LRATE_BASE',
                        default=0.002, type=float)
    parser.add_argument('--D_lrate_base', help='D learning rate.', metavar='D_LRATE_BASE',
                        default=0.002, type=float)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print('Error: dataset root directory does not exist.')
        sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
