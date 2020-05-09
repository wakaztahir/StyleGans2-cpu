#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_training_vc2.py
# --- Creation Date: 24-04-2020
# --- Last Modified: Sat 09 May 2020 15:47:29 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Run training file for variation consistency related networks (v2) use.
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


def run(dataset, data_dir, result_dir, config_id, num_gpus, total_kimg, gamma,
        mirror_augment, metrics, resume_pkl,
        I_fmap_base=8, G_fmap_base=8, D_fmap_base=9,
        fmap_decay=0.15, D_lambda=1, C_lambda=1, cls_alpha=0,
        n_samples_per=10, module_list=None, model_type='vc_gan2',
        epsilon_loss=3, random_eps=False, latent_type='uniform',
        delta_type='onedim', connect_mode='concat', batch_size=32, batch_per_gpu=16,
        return_atts=False, random_seed=1000,
        module_I_list=None, module_D_list=None,
        fmap_min=16, fmap_max=512,
        G_nf_scale=4, I_nf_scale=4, D_nf_scale=4, outlier_detector=False,
        gen_atts_in_D=False, no_atts_in_D=False, att_lambda=0,
        dlatent_size=24, arch='resnet'):
    # print('module_list:', module_list)
    train = EasyDict(run_func_name='training.training_loop_vc2.training_loop_vc2'
                     )  # Options for training loop.

    D_global_size = 0
    if not(module_list is None):
        module_list = _str_to_list(module_list)
        key_ls, size_ls, count_dlatent_size = split_module_names(module_list)
        for i, key in enumerate(key_ls):
            if key.startswith('D_global') or key.startswith('D_nocond_global'):
                D_global_size += size_ls[i]
    else:
        count_dlatent_size = dlatent_size

    if not(module_I_list is None):
        D_global_I_size = 0
        module_I_list = _str_to_list(module_I_list)
        key_I_ls, size_I_ls, count_dlatent_I_size = split_module_names(module_I_list)
        for i, key in enumerate(key_I_ls):
            if key.startswith('D_global') or key.startswith('D_nocond_global'):
                D_global_I_size += size_I_ls[i]
    if not(module_D_list is None):
        D_global_D_size = 0
        module_D_list = _str_to_list(module_D_list)
        key_D_ls, size_D_ls, count_dlatent_D_size = split_module_names(module_D_list)
        for i, key in enumerate(key_D_ls):
            if key.startswith('D_global') or key.startswith('D_nocond_global'):
                D_global_D_size += size_D_ls[i]

    if model_type == 'info_gan':
        G = EasyDict(func_name='training.vc_networks2.G_main_vc2',
            synthesis_func='G_synthesis_modular_vc2',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size, D_global_size=D_global_size,
            module_list=module_list, use_noise=True, G_nf_scale=G_nf_scale)  # Options for generator network.
        I = EasyDict(func_name='training.info_gan_networks.info_gan_body',
                     dlatent_size=count_dlatent_size,
                     D_global_size=D_global_size,
                     fmap_min=fmap_min, fmap_max=fmap_max)
        D = EasyDict(func_name='training.info_gan_networks.D_info_gan_stylegan2',
            fmap_min=fmap_min, fmap_max=fmap_max)  # Options for discriminator network.
        I_info = EasyDict()
        desc = 'info_gan_net'
    elif model_type == 'vc2_info_gan':
        G = EasyDict(
            func_name='training.vc_networks2.G_main_vc2',
            synthesis_func='G_synthesis_modular_vc2',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size, D_global_size=D_global_size,
            module_list=module_list, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale
        )  # Options for generator network.
        D = EasyDict(func_name='training.vc_networks2.D_info_modular_vc2',
                     dlatent_size=count_dlatent_D_size, D_global_size=D_global_D_size,
                     fmap_min=fmap_min, fmap_max=fmap_max,
                     connect_mode=connect_mode, module_D_list=module_D_list,
                     gen_atts_in_D=gen_atts_in_D,
                     no_atts_in_D=no_atts_in_D,
                     D_nf_scale=D_nf_scale)
        I = EasyDict()
        I_info = EasyDict()
        desc = 'vc2_info_gan_net'
    elif model_type == 'vc2_gan':
        G = EasyDict(
            func_name='training.vc_networks2.G_main_vc2',
            synthesis_func='G_synthesis_modular_vc2',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size, D_global_size=D_global_size,
            module_list=module_list, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale
        )  # Options for generator network.
        I = EasyDict(func_name='training.vc_networks2.vc2_head',
                     dlatent_size=count_dlatent_size, D_global_size=D_global_size,
                     fmap_min=fmap_min, fmap_max=fmap_max,
                     connect_mode=connect_mode)
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
            fmap_min=fmap_min, fmap_max=fmap_max)  # Options for discriminator network.
        I_info = EasyDict()
        desc = 'vc2_gan'
    elif model_type == 'vc2_gan_style2':
        G = EasyDict(
            func_name='training.vc_networks2.G_main_vc2',
            synthesis_func='G_synthesis_stylegan2_vc2',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay,
            latent_size=dlatent_size, architecture=arch,
            dlatent_size=count_dlatent_size, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale
        )  # Options for generator network.
        I = EasyDict(func_name='training.vc_networks2.vc2_head',
                     dlatent_size=count_dlatent_size, D_global_size=D_global_size,
                     fmap_min=fmap_min, fmap_max=fmap_max,
                     connect_mode=connect_mode)
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
            fmap_min=fmap_min, fmap_max=fmap_max)  # Options for discriminator network.
        I_info = EasyDict()
        desc = 'vc2_gan_style2'
    elif model_type == 'vc2_gan_style2_noI':
        G = EasyDict(
            func_name='training.vc_networks2.G_main_vc2',
            synthesis_func='G_synthesis_stylegan2_vc2',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay,
            latent_size=dlatent_size, architecture=arch,
            dlatent_size=count_dlatent_size, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale
        )  # Options for generator network.
        I = EasyDict()
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
            fmap_min=fmap_min, fmap_max=fmap_max)  # Options for discriminator network.
        I_info = EasyDict()
        desc = 'vc2_gan_style2'
    elif model_type == 'vc2_gan_own_I':
        G = EasyDict(
            func_name='training.vc_networks2.G_main_vc2',
            synthesis_func='G_synthesis_modular_vc2',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size, D_global_size=D_global_size,
            module_list=module_list, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale
        )  # Options for generator network.
        I = EasyDict(func_name='training.vc_networks2.I_modular_vc2',
                     dlatent_size=count_dlatent_I_size, D_global_size=D_global_I_size,
                     fmap_min=fmap_min, fmap_max=fmap_max,
                     connect_mode=connect_mode, module_I_list=module_I_list,
                     I_nf_scale=I_nf_scale)
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
            fmap_min=fmap_min, fmap_max=fmap_max)  # Options for discriminator network.
        I_info = EasyDict()
        desc = 'vc2_gan_ownI'
    elif model_type == 'vc2_gan_own_ID':
        G = EasyDict(
            func_name='training.vc_networks2.G_main_vc2',
            synthesis_func='G_synthesis_modular_vc2',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size, D_global_size=D_global_size,
            module_list=module_list, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale
        )  # Options for generator network.
        I = EasyDict(func_name='training.vc_networks2.I_modular_vc2',
                     dlatent_size=count_dlatent_I_size, D_global_size=D_global_I_size,
                     fmap_min=fmap_min, fmap_max=fmap_max,
                     connect_mode=connect_mode, module_I_list=module_I_list,
                     I_nf_scale=I_nf_scale)
        D = EasyDict(func_name='training.vc_networks2.D_modular_vc2',
                     dlatent_size=count_dlatent_D_size, D_global_size=D_global_D_size,
                     fmap_min=fmap_min, fmap_max=fmap_max,
                     connect_mode=connect_mode, module_D_list=module_D_list,
                     D_nf_scale=D_nf_scale)
        I_info = EasyDict()
        desc = 'vc2_gan_ownID'
    elif model_type == 'vc2_gan_noI':
        G = EasyDict(
            func_name='training.vc_networks2.G_main_vc2',
            synthesis_func='G_synthesis_modular_vc2',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size, D_global_size=D_global_size,
            module_list=module_list, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale
        )  # Options for generator network.
        I = EasyDict()
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
            fmap_min=fmap_min, fmap_max=fmap_max)  # Options for discriminator network.
        I_info = EasyDict()
        desc = 'vc2_gan_noI'
    else:
        raise ValueError('Not supported model tyle: ' + model_type)

    G_opt = EasyDict(beta1=0.0, beta2=0.99,
                     epsilon=1e-8)  # Options for generator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99,
                     epsilon=1e-8)  # Options for discriminator optimizer.
    if model_type == 'info_gan':
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns_info_gan',
            D_global_size=D_global_size, D_lambda=D_lambda, C_lambda=C_lambda,
            latent_type=latent_type)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss.D_logistic_r1_info_gan',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for discriminator loss.
    elif model_type == 'vc2_info_gan':
        G_loss = EasyDict(func_name='training.loss_vc2.G_logistic_ns_vc2_info_gan',
            D_global_size=D_global_size, C_lambda=C_lambda,
            epsilon=epsilon_loss, random_eps=random_eps, latent_type=latent_type,
            delta_type=delta_type, outlier_detector=outlier_detector,
            gen_atts_in_D=gen_atts_in_D, att_lambda=att_lambda)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss_vc2.D_logistic_r1_vc2_info_gan',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for discriminator loss.
    elif model_type == 'vc2_gan' or model_type == 'vc2_gan_style2':
        G_loss = EasyDict(func_name='training.loss_vc2.G_logistic_ns_vc2',
            D_global_size=D_global_size, C_lambda=C_lambda,
            epsilon=epsilon_loss, random_eps=random_eps, latent_type=latent_type,
            delta_type=delta_type)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss_vc2.D_logistic_r1_vc2',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for discriminator loss.
    elif model_type == 'vc2_gan_own_I' or model_type == 'vc2_gan_own_ID':
        G_loss = EasyDict(func_name='training.loss_vc2.G_logistic_ns_vc2',
            D_global_size=D_global_size, C_lambda=C_lambda,
            epsilon=epsilon_loss, random_eps=random_eps, latent_type=latent_type,
            delta_type=delta_type, own_I=True)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss_vc2.D_logistic_r1_vc2',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for discriminator loss.
    elif model_type == 'vc2_gan_noI' or model_type == 'vc2_gan_style2_noI':
        G_loss = EasyDict(func_name='training.loss_vc2.G_logistic_ns',
                          latent_type=latent_type)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss_vc2.D_logistic_r1_vc2',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for discriminator loss.

    sched = EasyDict()  # Options for TrainingSchedule.
    grid = EasyDict(size='1080p', layout='random')  # Options for setup_snapshot_image_grid().
    sc = dnnlib.SubmitConfig()  # Options for dnnlib.submit_run().
    # tf_config = {'rnd.np_random_seed': 1000}  # Options for tflib.init_tf().
    tf_config = {'rnd.np_random_seed': random_seed}  # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    sched.G_lrate_base = sched.D_lrate_base = 0.002
    sched.minibatch_size_base = batch_size
    sched.minibatch_gpu_base = batch_per_gpu
    D_loss.gamma = 10
    metrics = [metric_defaults[x] for x in metrics]

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset, max_label_size='full')

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus
    desc += '-' + config_id

    # Configs A-E: Shrink networks to match original StyleGAN.
    # I.fmap_base = 2 << 8
    # G.fmap_base = 2 << 8
    # D.fmap_base = 2 << 9
    I.fmap_base = 2 << I_fmap_base
    G.fmap_base = 2 << G_fmap_base
    D.fmap_base = 2 << D_fmap_base

    # Config E: Set gamma to 100 and override G & D architecture.
    # D_loss.gamma = 100

    if gamma is not None:
        D_loss.gamma = gamma

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, I_args=I, I_info_args=I_info, G_opt_args=G_opt, D_opt_args=D_opt,
                  G_loss_args=G_loss, D_loss_args=D_loss,
                  use_info_gan=(model_type == 'info_gan'),
                  use_vc_head=(model_type == 'vc2_gan' or
                               model_type == 'vc2_gan_own_I' or
                               model_type == 'vc2_gan_own_ID' or
                               model_type=='vc2_gan_style2'),
                  use_vc2_info_gan=(model_type == 'vc2_info_gan'),
                  traversal_grid=True, return_atts=return_atts)
    n_continuous = 0
    if not(module_list is None):
        for i, key in enumerate(key_ls):
            m_name = key.split('-')[0]
            if (m_name in LATENT_MODULES) and (not m_name == 'D_global'):
                n_continuous += size_ls[i]
    else:
        n_continuous = dlatent_size

    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics,
                  tf_config=tf_config, resume_pkl=resume_pkl, n_discrete=D_global_size,
                  n_continuous=n_continuous, n_samples_per=n_samples_per)
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
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--config', help='Training config (default: %(default)s)',
                        default='config-e', dest='config_id', metavar='CONFIG')
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)',
                        default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg',
        help='Training length in thousands of images (default: %(default)s)',
        metavar='KIMG', default=25000, type=int)
    parser.add_argument('--gamma',
        help='R1 regularization weight (default is config dependent)',
        default=None, type=float)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)',
                        default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument(
        '--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)',
        default='None', type=_parse_comma_sep)
    parser.add_argument('--model_type', help='Type of model to train', default='vc2_gan',
                        type=str, metavar='MODEL_TYPE', choices=['info_gan', 'vc2_gan', 'vc2_gan_noI',
                                                                 'vc2_gan_own_I', 'vc2_gan_own_ID',
                                                                 'vc2_info_gan', 'vc2_gan_style2',
                                                                 'vc2_gan_style2_noI'])
    parser.add_argument('--resume_pkl', help='Continue training using pretrained pkl.',
                        default=None, metavar='RESUME_PKL', type=str)
    parser.add_argument('--n_samples_per', help='Number of samples for each line in traversal (default: %(default)s)',
        metavar='N_SHOWN_SAMPLES_PER_LINE', default=10, type=int)
    parser.add_argument('--module_list', help='Module list for modular network.',
                        default=None, metavar='MODULE_LIST', type=str)
    parser.add_argument('--batch_size', help='N batch.',
                        metavar='N_BATCH', default=32, type=int)
    parser.add_argument('--batch_per_gpu', help='N batch per gpu.',
                        metavar='N_BATCH_PER_GPU', default=16, type=int)
    parser.add_argument('--D_lambda', help='Discrete lambda for INFO-GAN and VC-GAN.',
                        metavar='D_LAMBDA', default=1, type=float)
    parser.add_argument('--C_lambda', help='Continuous lambda for INFO-GAN and VC-GAN.',
                        metavar='C_LAMBDA', default=1, type=float)
    parser.add_argument('--cls_alpha', help='Classification hyper in VC-GAN.',
                        metavar='CLS_ALPHA', default=0, type=float)
    parser.add_argument('--epsilon_loss', help='Continuous lambda for INFO-GAN and VC-GAN.',
                        metavar='EPSILON_LOSS', default=0.4, type=float)
    parser.add_argument('--latent_type', help='What type of latent priori to use.',
                        metavar='LATENT_TYPE', default='uniform', choices=['uniform', 'normal', 'trunc_normal'], type=str)
    parser.add_argument('--random_eps',
        help='If use random epsilon in vc_gan_with_vc_head loss.',
        default=False, metavar='RANDOM_EPS', type=_str_to_bool)
    parser.add_argument('--delta_type', help='What type of delta use.',
                        metavar='DELTA_TYPE', default='onedim', choices=['onedim', 'fulldim'], type=str)
    parser.add_argument('--connect_mode', help='How fake1 and fake2 connected.',
                        default='concat', metavar='CONNECT_MODE', type=str)
    parser.add_argument('--fmap_decay', help='fmap decay for network building.',
                        metavar='FMAP_DECAY', default=0.15, type=float)
    parser.add_argument('--I_fmap_base', help='Fmap base for I.',
                        metavar='I_FMAP_BASE', default=8, type=int)
    parser.add_argument('--G_fmap_base', help='Fmap base for G.',
                        metavar='G_FMAP_BASE', default=8, type=int)
    parser.add_argument('--D_fmap_base', help='Fmap base for D.',
                        metavar='D_FMAP_BASE', default=9, type=int)
    parser.add_argument('--return_atts', help='If return attention maps.',
                        default=False, metavar='RETURN_ATTS', type=_str_to_bool)
    parser.add_argument('--random_seed', help='TF random seed.',
                        metavar='RANDOM_SEED', default=9, type=int)
    parser.add_argument('--module_I_list', help='Module list for I modular network.',
                        default=None, metavar='MODULE_I_LIST', type=str)
    parser.add_argument('--module_D_list', help='Module list for D modular network.',
                        default=None, metavar='MODULE_D_LIST', type=str)
    parser.add_argument('--fmap_min', help='FMAP min.',
                        metavar='FMAP_MIN', default=16, type=int)
    parser.add_argument('--fmap_max', help='FMAP max.',
                        metavar='FMAP_MAX', default=512, type=int)
    parser.add_argument('--G_nf_scale', help='N feature map scale for G.',
                        metavar='G_NF_SCALE', default=4, type=int)
    parser.add_argument('--I_nf_scale', help='N feature map scale for I.',
                        metavar='I_NF_SCALE', default=4, type=int)
    parser.add_argument('--D_nf_scale', help='N feature map scale for D.',
                        metavar='D_NF_SCALE', default=4, type=int)
    parser.add_argument('--outlier_detector', help='If use outlier detector instead of regressor.',
                        default=False, metavar='OUTLIER_DETECTOR', type=_str_to_bool)
    parser.add_argument('--gen_atts_in_D', help='If generate atts in D of vc2_infogan.',
                        default=False, metavar='GEN_ATTS_IN_D', type=_str_to_bool)
    parser.add_argument('--no_atts_in_D', help='If not use atts in D of vc2_infogan.',
                        default=False, metavar='NO_ATTS_IN_D', type=_str_to_bool)
    parser.add_argument('--att_lambda', help='ATT lambda of gen_atts in D for vc2_infogan loss.',
                        metavar='ATT_LAMBDA', default=0, type=float)
    parser.add_argument('--dlatent_size', help='Latent size. Used for vc2_gan_style2.',
                        metavar='DLATENT_SIZE', default=24, type=int)
    parser.add_argument('--arch', help='Architecture for vc2_gan_style2.',
                        metavar='ARCH', default='resnet', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print('Error: dataset root directory does not exist.')
        sys.exit(1)

    # if args.config_id not in _valid_configs:
        # print('Error: --config value must be one of: ',
              # ', '.join(_valid_configs))
        # sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
