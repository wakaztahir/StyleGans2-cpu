#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vae_so.py
# --- Creation Date: 06-12-2020
# --- Last Modified: Tue 08 Dec 2020 02:47:15 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
SO(n) vae losses.
"""
import numpy as np
import math
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
from training.utils import get_return_v
from training.loss_vae import sample_from_latent_distribution
from training.loss_vae import make_reconstruction_loss
from training.loss_vae import compute_gaussian_kl


def so_vae(E,
           G,
           opt,
           training_set,
           minibatch_size,
           reals,
           labels,
           hy_1p=0,
           latent_type='normal',
           recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var = get_return_v(
        E.get_output_for(reals, labels, is_training=True), 2)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)
    sampled = sample_from_latent_distribution(means, log_var)

    reconstructions, lie_groups_as_fm, _, _, lie_algs, lie_alg_basis, _, lie_vars = get_return_v(
        G.get_output_for(sampled, labels, is_training=True), 8)
    # lie_groups_as_fm: [b, lat_dim, mat_dim, mat_dim]
    # lie_algs: [b, lat_dim, mat_dim, mat_dim]
    # lie_alg_basis: [1, lat_dim, mat_dim, mat_dim]

    reconstruction_loss = make_reconstruction_loss(reals,
                                                   reconstructions,
                                                   recons_type=recons_type)
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)

    elbo = reconstruction_loss + kl_loss
    elbo = autosummary('Loss/so_vae_elbo', elbo)
    loss = elbo + hy_1p * tf.reduce_sum(lie_vars * lie_vars)
    loss = autosummary('Loss/so_vae_loss', loss)
    return loss
