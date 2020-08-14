#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: utils.py
# --- Creation Date: 14-08-2020
# --- Last Modified: Fri 14 Aug 2020 22:40:28 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Useful functions.
"""
import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
from training import misc

def save_atts(atts, filename, grid_size, drange, grid_fakes, n_samples_per):
    canvas = np.zeros([grid_fakes.shape[0], 1, grid_fakes.shape[2], grid_fakes.shape[3]])
    # atts: [b, n_latents, 1, res, res]

    for i in range(atts.shape[1]):
        att_sp = atts[:, i]  # [b, 1, x_h, x_w]
        grid_start_idx = i * n_samples_per
        canvas[grid_start_idx : grid_start_idx + n_samples_per] = att_sp[grid_start_idx : grid_start_idx + n_samples_per]

    # already_n_latents = 0
    # for i, att in enumerate(atts):
        # att_sp = att[-1]  # [b, n_latents, 1, x_h, x_w]
        # for j in range(att_sp.shape[1]):
            # att_sp_sub = att_sp[:, j]  # [b, 1, x_h, x_w]
            # grid_start_idx = already_n_latents * n_samples_per
            # canvas[grid_start_idx : grid_start_idx + n_samples_per] = att_sp_sub[grid_start_idx : grid_start_idx + n_samples_per]
            # already_n_latents += 1
    misc.save_image_grid(canvas,
                         filename,
                         drange=drange,
                         grid_size=grid_size)
    return

def add_outline(images, width=1):
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]
    for i in range(num):
        images[i, :, 0:width, :] = 255
        images[i, :, -width:, :] = 255
        images[i, :, :, 0:width] = 255
        images[i, :, :, -width:] = 255
    return images

def get_grid_latents(n_discrete, n_continuous, n_samples_per, G, grid_labels, topk_dims=None, latent_type='normal'):
    if n_discrete == 0:
        n_discrete = 1  # 0 discrete means 1 discrete
        real_has_discrete = False
    else:
        real_has_discrete = True
    grid_size = (n_samples_per, n_continuous * n_discrete)
    if latent_type == 'uniform':
        z = np.random.uniform(low=-2., high=2., size=(1, n_continuous))
    elif latent_type == 'normal':
        z = np.random.normal(size=(1, n_continuous))
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
        # z = np.random.randn(1, n_continuous)  # [minibatch, component-3]
    grid_latents = np.tile(z, (n_continuous * n_samples_per * n_discrete, 1))
    for i in range(n_discrete):
        for j in range(n_continuous):
            grid_latents[(i * n_continuous + j) *
                         n_samples_per:(i * n_continuous + j + 1) *
                         n_samples_per, j] = np.arange(
                             -2. + 4. / float(n_samples_per+1), 2., 4. / float(n_samples_per+1))
    if real_has_discrete:
        grid_discrete_ls = []
        for i in range(n_discrete):
            init_onehot = [0] * n_discrete
            init_onehot[i] = 1
            grid_discrete_ls.append(
                np.tile(np.array([init_onehot], dtype=np.float32),
                        (n_continuous * n_samples_per, 1)))
        grid_discrete = np.concatenate(grid_discrete_ls, axis=0)
        grid_latents = np.concatenate((grid_discrete, grid_latents), axis=1)
    grid_labels = np.tile(grid_labels[:1],
                          (n_discrete * n_continuous * n_samples_per, 1))
    if topk_dims is not None:
        grid_latents = np.reshape(grid_latents, [n_discrete, n_continuous, n_samples_per, -1])
        grid_latents = grid_latents[:, topk_dims]
        n_continuous = topk_dims
        grid_latents = np.reshape(grid_latents, [n_discrete * len(topk_dims) * n_samples_per, -1])
        grid_labels = np.tile(grid_labels[:1],
                              (n_discrete * len(topk_dims) * n_samples_per, 1))
        grid_size = (n_samples_per, len(topk_dims) * n_discrete)
    # grid_labels = np.tile(grid_labels[:1],
                          # (n_discrete * n_continuous * n_samples_per, 1))
    return grid_size, grid_latents, grid_labels