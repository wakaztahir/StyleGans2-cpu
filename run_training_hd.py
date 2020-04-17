#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_training_hd.py
# --- Creation Date: 06-04-2020
# --- Last Modified: Fri 17 Apr 2020 16:38:56 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
HD disentanglement models with pretrained G.
"""

import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults

#----------------------------------------------------------------------------

_valid_configs = [
    # Table 1
    'config-a', # Baseline StyleGAN
    'config-b', # + Weight demodulation
    'config-c', # + Lazy regularization
    'config-d', # + Path length regularization
    'config-e', # + No growing, new G & D arch.
    'config-f', # + Large networks (default)

    # Table 2
    'config-e-Gorig-Dorig',   'config-e-Gorig-Dresnet',   'config-e-Gorig-Dskip',
    'config-e-Gresnet-Dorig', 'config-e-Gresnet-Dresnet', 'config-e-Gresnet-Dskip',
    'config-e-Gskip-Dorig',   'config-e-Gskip-Dresnet',   'config-e-Gskip-Dskip',
]

#----------------------------------------------------------------------------

def run(dataset, data_dir, result_dir, config_id, num_gpus, total_kimg,
        mirror_augment, metrics, resume_G_pkl, n_batch=2, n_batch_per_gpu=1,
        D_global_size=0, C_global_size=10, model_type=False, latent_type='uniform',
        resume_pkl=None, n_samples_per=4, D_lambda=0, C_lambda=1,
        epsilon_in_loss=3, random_eps=True, M_lrmul=0.1, resolution_manual=1024,
        pretrained_type='with_stylegan2', traj_lambda=None, level_I_kimg=1000,
        use_level_training=False, resume_kimg=0, use_std_in_m=False, prior_latent_size=512,
        M_mapping_fmaps=512):
    train     = EasyDict(run_func_name='training.training_loop_hd.training_loop_hd') # Options for training loop with pretrained HD.
    if model_type == 'hd_hyperplane':
        M         = EasyDict(func_name='training.hd_networks.net_M_hyperplane',
                             C_global_size=C_global_size, D_global_size=D_global_size,
                             latent_size=prior_latent_size,
                             mapping_lrmul=M_lrmul, use_std_in_m=use_std_in_m)  # Options for dismapper network.
        I         = EasyDict(func_name='training.hd_networks.net_I',
                             C_global_size=C_global_size, D_global_size=D_global_size)  # Options for recognizor network.
    else:
        M         = EasyDict(func_name='training.hd_networks.net_M',
                             C_global_size=C_global_size, D_global_size=D_global_size,
                             latent_size=prior_latent_size,
                             mapping_fmaps=M_mapping_fmaps,
                             mapping_lrmul=M_lrmul, use_std_in_m=use_std_in_m)  # Options for dismapper network.
        I         = EasyDict(func_name='training.hd_networks.net_I',
                             C_global_size=C_global_size, D_global_size=D_global_size)  # Options for recognizor network.
    if model_type == 'hd_dis_model_with_cls':
        I_info = EasyDict(func_name='training.hd_networks.net_I_info',
                          C_global_size=C_global_size, D_global_size=D_global_size)
    else:
        I_info = EasyDict()
    I_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
    I_loss    = EasyDict(func_name='training.loss_hd.IandM_loss', latent_type=latent_type,
                         D_global_size=D_global_size,
                         C_global_size=C_global_size,
                         D_lambda=D_lambda, C_lambda=C_lambda,
                         epsilon=epsilon_in_loss, random_eps=random_eps,
                         traj_lambda=traj_lambda, resolution_manual=resolution_manual,
                         use_std_in_m=use_std_in_m)              # Options for discriminator loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='1080p', layout='random')                           # Options for setup_snapshot_image_grid().
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    sched.I_lrate_base = 0.002
    sched.minibatch_size_base = n_batch
    sched.minibatch_gpu_base = n_batch_per_gpu
    metrics = [metric_defaults[x] for x in metrics]
    desc = 'hd_disentanglement'

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset)

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    assert config_id in _valid_configs
    desc += '-' + config_id

    # Configs A-E: Shrink networks to match original StyleGAN.
    if config_id != 'config-f':
        I.fmap_base = 8 << 10

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(I_args=I, M_args=M, I_opt_args=I_opt,
                  I_loss_args=I_loss, resume_G_pkl=resume_G_pkl)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid,
                  use_hd_with_cls=(model_type == 'hd_dis_model_with_cls'),
                  metric_arg_list=metrics, tf_config=tf_config,
                  resume_pkl=resume_pkl, n_discrete=D_global_size,
                  n_continuous=C_global_size, n_samples_per=n_samples_per,
                  resolution_manual=resolution_manual, pretrained_type=pretrained_type,
                  level_I_kimg=level_I_kimg, use_level_training=use_level_training,
                  resume_kimg=resume_kimg, use_std_in_m=use_std_in_m,
                  prior_latent_size=prior_latent_size)
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

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

_examples = '''examples:

  # Train StyleGAN2 using the FFHQ dataset
  python %(prog)s --num-gpus=8 --data-dir=~/datasets --config=config-f --dataset=ffhq --mirror-augment=true

valid configs:

  ''' + ', '.join(_valid_configs) + '''

valid metrics:

  ''' + ', '.join(sorted([x for x in metric_defaults.keys()])) + '''

'''

def main():
    parser = argparse.ArgumentParser(
        description='Train StyleGAN2.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--config', help='Training config (default: %(default)s)', default='config-f', required=True, dest='config_id', metavar='CONFIG')
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=25000, type=int)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)', default='None', type=_parse_comma_sep)
    parser.add_argument('--model_type', help='Type of model to train', default='hd_dis_model',
                        type=str, metavar='MODEL_TYPE', choices=['hd_dis_model', 'hd_dis_model_with_cls', 'hd_hyperplane'])
    parser.add_argument('--D_global_size', help='Number of discrete latents',
                        metavar='D_GLOBAL_SIZE', default=0, type=int)
    parser.add_argument('--C_global_size', help='Number of continuous latents',
                        metavar='C_GLOBAL_SIZE', default=0, type=int)
    parser.add_argument('--D_lambda', help='Discrete lambda for INFO-GAN and HD-GAN.',
                        metavar='D_LAMBDA', default=1, type=float)
    parser.add_argument('--C_lambda', help='Continuous lambda for INFO-GAN and HD-GAN.',
                        metavar='C_LAMBDA', default=1, type=float)
    parser.add_argument('--latent_type', help='What type of latent priori to use.',
                        metavar='LATENT_TYPE',
                        default='uniform', choices=['uniform', 'normal', 'trunc_normal'], type=str)
    parser.add_argument('--resume_pkl', help='Continue training using pretrained pkl.',
                        default=None, metavar='RESUME_PKL', type=str)
    parser.add_argument('--n_samples_per',
                        help='Number of samples for each line in traversal (default: %(default)s)',
                        metavar='N_SHOWN_SAMPLES_PER_LINE', default=4, type=int)
    parser.add_argument('--resume_G_pkl', help='Pretrained G pkl.',
                        default=None, metavar='RESUME_G_PKL', type=str)
    parser.add_argument('--random_eps', help='If use random epsilon in loss.',
                        default=True, metavar='RANDOM_EPS', type=_str_to_bool)
    parser.add_argument('--epsilon_in_loss', help='The epsilon value used in loss.',
                        metavar='EPSILON_IN_LOSS', default=3, type=float)
    parser.add_argument('--M_lrmul', help='Learning rate multiplier in M net.',
                        metavar='M_LRMUL', default=0.1, type=float)
    parser.add_argument('--resolution_manual', help='Resolution of generated images.',
                        metavar='RESOLUTION_MANUAL', default=1024, type=int)
    parser.add_argument('--pretrained_type', help='Pretrained type for G.',
                        metavar='PRETRAINED_TYPE', default='with_stylegan2', type=str)
    parser.add_argument('--traj_lambda', help='Hyperparam for prior trajectory regularization.',
                        metavar='TRAJ_LAMBDA', default=None, type=float)
    parser.add_argument('--level_I_kimg', help='Number of kimg of tick for I_level training.',
                        metavar='LEVEL_I_KIMG', default=1000, type=int)
    parser.add_argument('--n_batch', help='N batch.',
                        metavar='N_BATCH', default=2, type=int)
    parser.add_argument('--n_batch_per_gpu', help='N batch per gpu.',
                        metavar='N_BATCH_PER_GPU', default=1, type=int)
    parser.add_argument('--resume_kimg', help='K number of imgs have been trained.',
                        metavar='RESUME_KIMG', default=0, type=int)
    parser.add_argument('--use_level_training', help='If use level training strategy.',
                        default=False, metavar='USE_LEVEL_TRAINING', type=_str_to_bool)
    parser.add_argument('--use_std_in_m', help='If output prior std in M net.',
                        default=False, metavar='USE_STD_IN_M', type=_str_to_bool)
    parser.add_argument('--prior_latent_size', help='Size of prior latent space.',
                        metavar='PRIOR_LATENTS_SIZE', default=512, type=int)
    parser.add_argument('--M_mapping_fmaps', help='M mapping net fmaps size.',
                        metavar='M_MAPPING_FMAPS', default=512, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)

    if args.config_id not in _valid_configs:
        print ('Error: --config value must be one of: ', ', '.join(_valid_configs))
        sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print ('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

