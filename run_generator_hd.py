#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_generator_hd.py
# --- Creation Date: 15-04-2020
# --- Last Modified: Wed 15 Apr 2020 18:32:01 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Generator script for HD disentanglement models.
"""

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import pretrained_networks
from training import misc
from training.training_loop_dsp import get_grid_latents

#----------------------------------------------------------------------------


def generate_images(network_pkl,
                    network_G_pkl,
                    n_imgs,
                    model_type,
                    n_discrete,
                    n_continuous,
                    use_std_in_m=None,
                    latent_type='uniform',
                    n_samples_per=10):
    print('Loading networks from "%s"...' % network_pkl)
    tflib.init_tf()
    if model_type == 'hd_dis_model_with_cls':
        # _G, _D, I, Gs = misc.load_pkl(network_pkl)
        I, M, Is, I_info = misc.load_pkl(network_pkl)
    else:
        # _G, _D, Gs = misc.load_pkl(network_pkl)
        I, M, Is = misc.load_pkl(network_pkl)

    # Load pretrained GAN
    _G, _D, Gs = misc.load_pkl(network_G_pkl)

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                      nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    for idx in range(n_imgs):
        print('Generating image %d/%d ...' % (idx, n_imgs))

        if n_discrete == 0:
            grid_labels = np.zeros([n_continuous * n_samples_per, 0],
                                   dtype=np.float32)
        else:
            grid_labels = np.zeros(
                [n_discrete * n_continuous * n_samples_per, 0],
                dtype=np.float32)

        grid_size, grid_latents, grid_labels = get_grid_latents(
            n_discrete, n_continuous, n_samples_per, _G, grid_labels, 
            latent_type=latent_type)
        prior_traj_latents = M.run(grid_latents,
                            is_validation=True,
                            minibatch_size=4)
        if use_std_in_m is not None:
            prior_traj_latents = prior_traj_latents[:, :prior_traj_latents.shape[1]//2]
        grid_fakes = Gs.run(prior_traj_latents,
                            grid_labels,
                            is_validation=True,
                            minibatch_size=4,
                            randomize_noise=False)
        print(grid_fakes.shape)
        misc.save_image_grid(grid_fakes,
                             dnnlib.make_run_dir_path('img_%04d.png' % idx),
                             drange=[-1, 1],
                             grid_size=grid_size)
        frames = []
        grid_fakes = np.reshape(grid_fakes,
                                [n_continuous, n_samples_per,
                                 grid_fakes.shape[1], grid_fakes.shape[2], grid_fakes.shape[3]])
        for i in range(n_samples_per):
            to_concat = [grid_fakes[j, i] for j in range(n_continuous)]
            to_concat = tuple(to_concat)
            grid_fake_pil = misc.convert_to_pil_image(np.concatenate(to_concat, axis=2))
            frames.append(grid_fake_pil)
        frames[0].save(dnnlib.make_run_dir_path('latents_trav_%04d.gif' % idx),
                       format='GIF',
                       append_images=frames[1:],
                       save_all=True, duration=100, loop=0)

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate images traversals
  python %(prog)s --network_pkl=results/hd_gan.pkl --network_G_pkl=results_pretrained/gan.pkl --n_imgs=5 --result_dir ./results
'''


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='''HD GAN generator.

Run 'python %(prog)s --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--network_pkl',
                        help='Network pickle filename',
                        required=True)
    parser.add_argument('--network_G_pkl',
                        help='Network Generator pickle filename',
                        required=True)
    parser.add_argument('--n_imgs',
                        type=int,
                        help='Number of images to generate',
                        required=True)
    parser.add_argument('--n_discrete',
                        type=int,
                        help='Number of discrete latents',
                        default=0)
    parser.add_argument('--n_continuous',
                        type=int,
                        help='Number of continuous latents',
                        default=14)
    parser.add_argument('--n_samples_per',
                        type=int,
                        help='Number of samples per row',
                        default=10)
    parser.add_argument('--model_type',
                        type=str,
                        help='Which model is this pkl',
                        default='hd_dis_model',
                        choices=['hd_dis_model', 'hd_dis_model_with_cls'])
    parser.add_argument('--use_std_in_m',
                        type=int,
                        help='If use std in M net',
                        default=None)
    parser.add_argument('--latent_type',
                        type=str,
                        help='Latent type',
                        default='uniform',
                        choices=['uniform', 'normal'])
    parser.add_argument(
        '--result-dir',
        help='Root directory for run results (default: %(default)s)',
        default='results',
        metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')

    dnnlib.submit_run(sc, 'run_generator_hd.generate_images', **kwargs)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
