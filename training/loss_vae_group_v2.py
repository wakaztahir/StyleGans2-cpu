#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vae_group_v2.py
# --- Creation Date: 27-09-2020
# --- Last Modified: Sun 27 Sep 2020 18:44:21 AEST
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


def make_lie_group_act_loss(group_feats_E, group_feats_G,
                            lie_alg_feats, lie_alg_basis, minibatch_size, hy_rec,
                            hy_dcp, hy_gmat, hy_hes, hy_lin, hy_ncut):
    mat_dim = group_feats_G.get_shape().as_list()[1]
    group_feats_G_ori = group_feats_G[:minibatch_size]
    group_feats_G_sum = group_feats_G[minibatch_size:]

    group_feats_G_mul = tf.matmul(
        group_feats_G[:minibatch_size // 2],
        group_feats_G[minibatch_size // 2:minibatch_size])

    lie_alg_basis_square = lie_alg_basis * lie_alg_basis
    # [1, lat_dim, mat_dim, mat_dim]
    _, lat_dim, mat_dim, _ = lie_alg_basis.get_shape().as_list()
    lie_alg_basis_col = tf.reshape(lie_alg_basis, [lat_dim, 1, mat_dim, mat_dim])
    lie_alg_basis_mul = tf.matmul(lie_alg_basis, lie_alg_basis_col)
    lie_alg_basis_mask = 1. - tf.eye(lat_dim, dtype=lie_alg_basis_mul.dtype)[:, :, tf.newaxis, tf.newaxis]
    lie_alg_basis_mul = lie_alg_basis_mul * lie_alg_basis_mask

    gmat_loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(group_feats_G_mul - group_feats_G_sum), axis=[1, 2]))
    gmat_loss = autosummary('Loss/lie_vae_gmat_loss', gmat_loss)
    lin_loss = tf.reduce_mean(tf.reduce_sum(lie_alg_basis_square, axis=[2, 3]))
    lin_loss = autosummary('Loss/lie_vae_linear_loss', lin_loss)
    hessian_loss = tf.reduce_mean(tf.reduce_sum(tf.square(lie_alg_basis_mul), axis=[2, 3]))
    hessian_loss = autosummary('Loss/lie_vae_hessian_loss', hessian_loss)

    loss = hy_gmat * gmat_loss + hy_hes * hessian_loss + hy_lin * lin_loss
    return loss


def group_act_vae(E,
                  G,
                  opt,
                  training_set,
                  minibatch_size,
                  reals,
                  labels,
                  latent_type='normal',
                  hy_dcp=1,
                  hy_gmat=1,
                  hy_hes=0,
                  hy_lin=0,
                  hy_ncut=1,
                  hy_rec=1,
                  recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var, group_feats_E = get_return_v(
        E.get_output_for(reals, labels, is_training=True), 3)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)

    mat_dim = int(math.sqrt(group_feats_E.get_shape().as_list()[1]))
    assert mat_dim * mat_dim == group_feats_E.get_shape().as_list()[1]
    group_feats_E = tf.reshape(group_feats_E,
                               [minibatch_size, mat_dim, mat_dim])

    sampled = sample_from_latent_distribution(means, log_var)

    sampled_sum = sampled[:minibatch_size // 2] + sampled[minibatch_size // 2:]
    sampled_all = tf.concat([sampled, sampled_sum], axis=0)
    labels_all = tf.concat([labels, labels[:minibatch_size // 2]], axis=0)

    reconstructions, group_feats_G, _, _, lie_alg_feats, lie_alg_basis = get_return_v(
        G.get_output_for(sampled_all, labels_all, is_training=True), 6)
    lie_group_loss = make_lie_group_act_loss(group_feats_E, group_feats_G,
                                             lie_alg_feats, lie_alg_basis, minibatch_size, hy_rec,
                                             hy_dcp, hy_gmat, hy_hes, hy_lin, hy_ncut)
    lie_group_loss = autosummary('Loss/lie_group_loss', lie_group_loss)

    reconstruction_loss = make_reconstruction_loss(
        reals, reconstructions[:minibatch_size], recons_type=recons_type)
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)

    elbo = reconstruction_loss + kl_loss
    elbo = autosummary('Loss/lie_vae_elbo', elbo)
    loss = elbo + lie_group_loss

    loss = autosummary('Loss/lie_vae_loss', loss)
    return loss
