#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vae_group_v3.py
# --- Creation Date: 14-12-2020
# --- Last Modified: Wed 16 Dec 2020 16:12:14 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss GroupVAE v2.
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
from training.loss_vae import split_latents


def make_lie_group_norm_loss(group_feats_G,
                             lie_alg_feats,
                             lie_alg_basis_norm,
                             minibatch_size,
                             hy_hes,
                             hy_commute=0):
    '''
    lie_alg_basis_norm: [1, lat_dim, mat_dim, mat_dim]
    '''
    _, lat_dim, mat_dim, _ = lie_alg_basis_norm.get_shape().as_list()
    lie_alg_basis_norm_col = tf.reshape(lie_alg_basis_norm,
                                        [lat_dim, 1, mat_dim, mat_dim])
    lie_alg_basis_outer_mul = tf.matmul(
        lie_alg_basis_norm,
        lie_alg_basis_norm_col)  # [lat_dim, lat_dim, mat_dim, mat_dim]
    hessian_mask = 1. - tf.eye(
        lat_dim,
        dtype=lie_alg_basis_outer_mul.dtype)[:, :, tf.newaxis, tf.newaxis]
    lie_alg_basis_mul_ij = lie_alg_basis_outer_mul * hessian_mask  # XY
    lie_alg_commutator = lie_alg_basis_mul_ij - tf.transpose(
        lie_alg_basis_mul_ij, [0, 1, 3, 2])
    loss = 0.
    hessian_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(lie_alg_basis_mul_ij), axis=[2, 3]))
    hessian_loss = autosummary('Loss/hessian', hessian_loss)
    hessian_loss *= hy_hes
    loss += hessian_loss
    if hy_commute > 0:
        print('using commute loss')
        commute_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(lie_alg_commutator), axis=[2, 3]))
        commute_loss = autosummary('Loss/commute', commute_loss)
        commute_loss *= hy_commute
        loss += commute_loss
    return loss


def group_norm_vae(E,
                   G,
                   opt,
                   training_set,
                   minibatch_size,
                   reals,
                   labels,
                   latent_type='normal',
                   hy_beta=1,
                   hy_hes=0,
                   hy_commute=0,
                   recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var = get_return_v(
        E.get_output_for(reals, labels, is_training=True), 2)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)

    sampled = sample_from_latent_distribution(means, log_var)

    reconstructions, group_feats_G, _, _, lie_alg_feats, lie_alg_basis_norm, _, lie_vars = get_return_v(
        G.get_output_for(sampled, labels, is_training=True), 8)
    lie_group_loss = make_lie_group_norm_loss(
        group_feats_G=group_feats_G,
        lie_alg_feats=lie_alg_feats,
        lie_alg_basis_norm=lie_alg_basis_norm,
        minibatch_size=minibatch_size,
        hy_hes=hy_hes,
        hy_commute=hy_commute)

    reconstruction_loss = make_reconstruction_loss(reals,
                                                   reconstructions,
                                                   recons_type=recons_type)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)

    elbo = reconstruction_loss + hy_beta * kl_loss
    elbo = autosummary('Loss/elbo', elbo)
    loss = elbo + lie_group_loss
    loss = autosummary('Loss/loss', loss)
    return loss
