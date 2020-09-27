#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vae_lie.py
# --- Creation Date: 21-09-2020
# --- Last Modified: Sun 27 Sep 2020 18:05:08 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss of LieVAE.
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


def make_lie_group_loss_with_split(group_feats_E, group_feats_G, lie_alg_feats, lie_alg_basis,
                        minibatch_size, hy_rec, hy_dcp, hy_hes, hy_lin,
                        hy_ncut):
    mat_dim = group_feats_G.get_shape().as_list()[1]
    gfeats_G = group_feats_G[:minibatch_size]
    gfeats_G_split_ls = [
        group_feats_G[(i + 1) * minibatch_size:(i + 2) * minibatch_size]
        for i in range(hy_ncut + 1)
    ]
    lie_alg_G_split_ls = [
        lie_alg_feats[(i + 1) * minibatch_size:(i + 2) * minibatch_size]
        for i in range(hy_ncut + 1)
    ]

    gfeats_G_split_mul = gfeats_G_split_ls[0]
    for i in range(1, hy_ncut + 1):
        gfeats_G_split_mul = tf.matmul(gfeats_G_split_mul,
                                       gfeats_G_split_ls[i])

    lie_alg_G_split_mul = lie_alg_G_split_ls[0]
    lie_alg_linear_G_split_mul = lie_alg_G_split_ls[0]
    for i in range(1, hy_ncut + 1):
        lie_alg_G_split_mul = tf.matmul(lie_alg_G_split_mul,
                                        lie_alg_G_split_ls[i])
        lie_alg_linear_G_split_mul = lie_alg_linear_G_split_mul * lie_alg_G_split_ls[
            i]

    # [1, lat_dim, mat_dim, mat_dim]
    _, lat_dim, mat_dim, _ = lie_alg_basis.get_shape().as_list()
    lie_alg_basis_col = tf.reshape(lie_alg_basis, [lat_dim, 1, mat_dim, mat_dim])
    lie_alg_basis_mul = tf.matmul(lie_alg_basis, lie_alg_basis_col)
    lie_alg_basis_mask = 1. - tf.eye(lat_dim, dtype=lie_alg_basis_mul.dtype)[:, :, tf.newaxis, tf.newaxis]
    lie_alg_basis_mul = lie_alg_basis_mul * lie_alg_basis_mask

    lie_alg_basis_linear = lie_alg_basis * lie_alg_basis_col
    lie_alg_basis_linear = lie_alg_basis_linear * lie_alg_basis_mask

    if group_feats_E is None:
        rec_loss = 0
    else:
        rec_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(group_feats_E - gfeats_G), axis=[1, 2]))
    # spl_loss = tf.reduce_mean(
        # tf.reduce_sum(tf.square(gfeats_G_split_mul - gfeats_G), axis=[1, 2]))
    spl_loss = tf.reduce_mean(tf.square(lie_alg_basis_mul - tf.transpose(lie_alg_basis_mul, perm=[1, 0, 2, 3])))
    # hessian_loss = tf.reduce_mean(
        # tf.reduce_sum(tf.square(lie_alg_G_split_mul), axis=[1, 2]))
    hessian_loss = tf.reduce_mean(tf.square(lie_alg_basis_mul))
    # linear_loss = tf.reduce_mean(
        # tf.reduce_sum(tf.square(lie_alg_linear_G_split_mul), axis=[1, 2]))
    linear_loss = tf.reduce_mean(tf.square(lie_alg_basis_linear))
    loss = hy_rec * rec_loss + hy_dcp * spl_loss + \
        hy_hes * hessian_loss + hy_lin * linear_loss
    return loss


def lie_vae_with_split(E,
            G,
            opt,
            training_set,
            minibatch_size,
            reals,
            labels,
            latent_type='normal',
            hy_dcp=1,
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

    sampled_split_ls = split_latents(sampled, minibatch_size, hy_ncut=hy_ncut)
    sampled_split = tf.concat(sampled_split_ls, axis=0)
    labels_split = tf.concat([labels] * len(sampled_split_ls), axis=0)

    sampled_all = tf.concat([sampled, sampled_split], axis=0)
    labels_all = tf.concat([labels, labels_split], axis=0)

    reconstructions, group_feats_G, _, _, lie_alg_feats, lie_alg_basis = get_return_v(
        G.get_output_for(sampled_all, labels_all, is_training=True), 6)
    lie_group_loss = make_lie_group_loss_with_split(group_feats_E, group_feats_G,
                                         lie_alg_feats, lie_alg_basis, minibatch_size, hy_rec,
                                         hy_dcp, hy_hes, hy_lin, hy_ncut)
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


def make_lie_group_loss_all(group_feats_E, group_feats_G, lie_alg_feats, lie_alg_basis,
                        minibatch_size, hy_rec, hy_dcp, hy_hes, hy_lin,
                        hy_ncut):
    mat_dim = group_feats_G.get_shape().as_list()[1]

    # [1, lat_dim, mat_dim, mat_dim]
    _, lat_dim, mat_dim, _ = lie_alg_basis.get_shape().as_list()
    lie_alg_basis_col = tf.reshape(lie_alg_basis, [lat_dim, 1, mat_dim, mat_dim])
    lie_alg_basis_mul = tf.matmul(lie_alg_basis, lie_alg_basis_col)
    lie_alg_basis_mask = 1. - tf.eye(lat_dim, dtype=lie_alg_basis_mul.dtype)[:, :, tf.newaxis, tf.newaxis]
    lie_alg_basis_mul = lie_alg_basis_mul * lie_alg_basis_mask

    lie_alg_basis_linear = lie_alg_basis * lie_alg_basis_col
    lie_alg_basis_linear = lie_alg_basis_linear * (1. - lie_alg_basis_mask)

    if group_feats_E is None:
        rec_loss = 0
    else:
        rec_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(group_feats_E - group_feats_G), axis=[1, 2]))
    rec_loss = autosummary('Loss/lie_vae_rec_loss', rec_loss)
    spl_loss = tf.reduce_sum(tf.square(lie_alg_basis_mul - tf.transpose(lie_alg_basis_mul, perm=[1, 0, 2, 3])))
    spl_loss = autosummary('Loss/lie_vae_spl_loss', spl_loss)
    hessian_loss = tf.reduce_sum(tf.square(lie_alg_basis_mul))
    hessian_loss = autosummary('Loss/lie_vae_hessian_loss', hessian_loss)
    linear_loss = tf.reduce_sum(tf.square(lie_alg_basis_linear))
    linear_loss = autosummary('Loss/lie_vae_linear_loss', linear_loss)
    loss = hy_rec * rec_loss + hy_dcp * spl_loss + \
        hy_hes * hessian_loss + hy_lin * linear_loss
    return loss


def make_lie_group_loss(group_feats_E, group_feats_G, lie_alg_feats, lie_alg_basis,
                        minibatch_size, hy_rec, hy_dcp, hy_hes, hy_lin,
                        hy_ncut):
    mat_dim = group_feats_G.get_shape().as_list()[1]

    # [1, lat_dim, mat_dim, mat_dim]
    _, lat_dim, mat_dim, _ = lie_alg_basis.get_shape().as_list()
    lie_alg_basis_col = tf.reshape(lie_alg_basis, [lat_dim, 1, mat_dim, mat_dim])
    lie_alg_basis_mul = tf.matmul(lie_alg_basis, lie_alg_basis_col)
    lie_alg_basis_mask = 1. - tf.eye(lat_dim, dtype=lie_alg_basis_mul.dtype)[:, :, tf.newaxis, tf.newaxis]
    lie_alg_basis_mul = lie_alg_basis_mul * lie_alg_basis_mask

    lie_alg_basis_linear = lie_alg_basis * lie_alg_basis_col
    lie_alg_basis_linear = lie_alg_basis_linear * (1. - lie_alg_basis_mask)

    hessian_loss = tf.reduce_sum(tf.square(lie_alg_basis_mul))
    hessian_loss = autosummary('Loss/lie_vae_hessian_loss', hessian_loss)
    linear_loss = tf.reduce_sum(tf.square(lie_alg_basis_linear))
    linear_loss = autosummary('Loss/lie_vae_linear_loss', linear_loss)
    loss = hy_hes * hessian_loss + hy_lin * linear_loss
    return loss


def lie_vae(E,
            G,
            opt,
            training_set,
            minibatch_size,
            reals,
            labels,
            latent_type='normal',
            hy_dcp=1,
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

    reconstructions, group_feats_G, _, _, lie_alg_feats, lie_alg_basis = get_return_v(
        G.get_output_for(sampled, labels, is_training=True), 6)
    lie_group_loss = make_lie_group_loss(group_feats_E, group_feats_G,
                                         lie_alg_feats, lie_alg_basis, minibatch_size, hy_rec,
                                         hy_dcp, hy_hes, hy_lin, hy_ncut)
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
