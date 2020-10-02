#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vae_group_v2.py
# --- Creation Date: 27-09-2020
# --- Last Modified: Fri 02 Oct 2020 18:58:03 AEST
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


def make_lie_group_act_loss(group_feats_E, group_feats_G, lie_alg_feats,
                            lie_alg_basis, act_points, minibatch_size, hy_rec,
                            hy_dcp, hy_gmat, hy_hes, hy_lin, hy_ncut, hessian_type):
    mat_dim = group_feats_G.get_shape().as_list()[1]
    mat_dim_E = group_feats_E.get_shape().as_list()[1]
    group_feats_G_ori = group_feats_G[:minibatch_size]
    group_feats_G_sum = group_feats_G[minibatch_size:minibatch_size + minibatch_size // 2]
    # gfeats_G_split_ls = [
        # group_feats_G[(i + 1) * minibatch_size + minibatch_size // 2:
                      # (i + 2) * minibatch_size + minibatch_size // 2]
        # for i in range(hy_ncut + 1)
    # ]
    # gfeats_G_split_mul = gfeats_G_split_ls[0]
    # for i in range(1, hy_ncut + 1):
        # gfeats_G_split_mul = tf.matmul(gfeats_G_split_mul,
                                       # gfeats_G_split_ls[i])

    group_feats_G_mul = tf.matmul(
        group_feats_G[:minibatch_size // 2],
        group_feats_G[minibatch_size // 2:minibatch_size])

    lie_alg_basis_square = lie_alg_basis * lie_alg_basis
    # [1, lat_dim, mat_dim, mat_dim]
    _, lat_dim, mat_dim, _ = lie_alg_basis.get_shape().as_list()
    lie_alg_basis_col = tf.reshape(lie_alg_basis,
                                   [lat_dim, 1, mat_dim, mat_dim])
    lie_alg_basis_mul = tf.matmul(lie_alg_basis, lie_alg_basis_col)
    lie_alg_basis_mask = 1. - tf.eye(
        lat_dim, dtype=lie_alg_basis_mul.dtype)[:, :, tf.newaxis, tf.newaxis]
    lie_alg_basis_mul = lie_alg_basis_mul * lie_alg_basis_mask

    # if mat_dim_E != mat_dim:
        # rec_loss = 0
    # else:
        # rec_loss = tf.reduce_mean(
            # tf.reduce_sum(tf.square(group_feats_E - group_feats_G_ori), axis=[1, 2]))
    # rec_loss = autosummary('Loss/lie_vae_rec_loss', rec_loss)


    gmat_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(group_feats_G_mul - group_feats_G_sum),
                      axis=[1, 2]))
    gmat_loss = autosummary('Loss/lie_vae_gmat_loss', gmat_loss)

    # spl_loss = tf.reduce_mean(
        # tf.reduce_sum(tf.square(gfeats_G_split_mul - group_feats_G_ori), axis=[1, 2]))
    # spl_loss = autosummary('Loss/lie_vae_spl_loss', spl_loss)

    lin_loss = tf.reduce_mean(tf.reduce_sum(lie_alg_basis_square, axis=[2, 3]))
    lin_loss = autosummary('Loss/lie_vae_linear_loss', lin_loss)
    if hessian_type == 'no_act_points':
        hessian_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(lie_alg_basis_mul), axis=[2, 3]))
    elif hessian_type == 'with_act_points':
        # act_points: [1, mat_dim, n_act_points]
        # lie_alg_basis_mul: [lat_dim, lat_dim, mat_dim, mat_dim]
        lie_act_mul = tf.matmul(lie_alg_basis_mul, act_points)
        # [lat_dim, lat_dim, mat_dim, n_act_points]
        # print('lie_act_mul.shape:', lie_act_mul.get_shape().as_list())
        hessian_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(lie_act_mul), axis=[2, 3]))
    else:
        raise ValueError('Not recognized hessian_type:', hessian_type)
    hessian_loss = autosummary('Loss/lie_vae_hessian_loss', hessian_loss)

    # loss = hy_gmat * gmat_loss + hy_dcp * spl_loss + hy_hes * hessian_loss + hy_lin * lin_loss
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
                  hy_dcp=0,
                  hy_gmat=0,
                  hy_hes=0,
                  hy_lin=0,
                  hy_ncut=0,
                  hy_rec=0,
                  hessian_type='no_act_points',
                  recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var, group_feats_E = get_return_v(
        E.get_output_for(reals, labels, is_training=True), 3)
    print('group_feats_E.shape:', group_feats_E.get_shape().as_list())
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)

    mat_dim = int(math.sqrt(group_feats_E.get_shape().as_list()[1]))
    assert mat_dim * mat_dim == group_feats_E.get_shape().as_list()[1]
    group_feats_E = tf.reshape(group_feats_E,
                               [minibatch_size, mat_dim, mat_dim])

    sampled = sample_from_latent_distribution(means, log_var)

    sampled_sum = sampled[:minibatch_size // 2] + sampled[minibatch_size // 2:]

    # sampled_split_ls = split_latents(sampled, minibatch_size, hy_ncut=hy_ncut)
    # sampled_split = tf.concat(sampled_split_ls, axis=0)
    # labels_split = tf.concat([labels] * len(sampled_split_ls), axis=0)

    # sampled_all = tf.concat([sampled, sampled_sum, sampled_split], axis=0)
    # labels_all = tf.concat([labels, labels[:minibatch_size // 2], labels_split], axis=0)

    sampled_all = tf.concat([sampled, sampled_sum], axis=0)
    labels_all = tf.concat([labels, labels[:minibatch_size // 2]], axis=0)

    reconstructions, group_feats_G, _, _, lie_alg_feats, lie_alg_basis, act_points = get_return_v(
        G.get_output_for(sampled_all, labels_all, is_training=True), 7)
    lie_group_loss = make_lie_group_act_loss(group_feats_E=group_feats_E,
                                             group_feats_G=group_feats_G,
                                             lie_alg_feats=lie_alg_feats,
                                             lie_alg_basis=lie_alg_basis,
                                             act_points=act_points,
                                             minibatch_size=minibatch_size,
                                             hy_rec=hy_rec,
                                             hy_dcp=hy_dcp,
                                             hy_gmat=hy_gmat,
                                             hy_hes=hy_hes,
                                             hy_lin=hy_lin,
                                             hy_ncut=hy_ncut,
                                             hessian_type=hessian_type)
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

def make_lie_group_act_spl_loss(group_feats_E, group_feats_G, lie_alg_feats,
                                lie_alg_basis, act_points, minibatch_size, hy_rec,
                                hy_dcp, hy_gmat, hy_hes, hy_lin, hy_ncut, hessian_type):
    mat_dim = group_feats_G.get_shape().as_list()[1]
    mat_dim_E = group_feats_E.get_shape().as_list()[1]
    group_feats_G_ori = group_feats_G[:minibatch_size]
    group_feats_G_sum = group_feats_G[minibatch_size:minibatch_size + minibatch_size // 2]

    group_feats_G_mul = tf.matmul(
        group_feats_G[:minibatch_size // 2],
        group_feats_G[minibatch_size // 2:minibatch_size])

    lie_alg_basis_square = lie_alg_basis * lie_alg_basis
    # [1, lat_dim, mat_dim, mat_dim]
    _, lat_dim, mat_dim, _ = lie_alg_basis.get_shape().as_list()
    lie_alg_basis_col = tf.reshape(lie_alg_basis,
                                   [lat_dim, 1, mat_dim, mat_dim])
    lie_alg_basis_mul = tf.matmul(lie_alg_basis, lie_alg_basis_col)
    lie_alg_basis_mask = 1. - tf.eye(
        lat_dim, dtype=lie_alg_basis_mul.dtype)[:, :, tf.newaxis, tf.newaxis]
    lie_alg_basis_mul = lie_alg_basis_mul * lie_alg_basis_mask

    # if mat_dim_E != mat_dim:
        # rec_loss = 0
    # else:
        # rec_loss = tf.reduce_mean(
            # tf.reduce_sum(tf.square(group_feats_E - group_feats_G_ori), axis=[1, 2]))
    # rec_loss = autosummary('Loss/lie_vae_rec_loss', rec_loss)


    gmat_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(group_feats_G_mul - group_feats_G_sum),
                      axis=[1, 2]))
    gmat_loss = autosummary('Loss/lie_vae_gmat_loss', gmat_loss)

    # spl_loss = tf.reduce_mean(
        # tf.reduce_sum(tf.square(gfeats_G_split_mul - group_feats_G_ori), axis=[1, 2]))
    # spl_loss = autosummary('Loss/lie_vae_spl_loss', spl_loss)

    lin_loss = tf.reduce_mean(tf.reduce_sum(lie_alg_basis_square, axis=[2, 3]))
    lin_loss = autosummary('Loss/lie_vae_linear_loss', lin_loss)
    if hessian_type == 'no_act_points':
        hessian_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(lie_alg_basis_mul), axis=[2, 3]))
    elif hessian_type == 'with_act_points':
        # act_points: [1, mat_dim, n_act_points]
        # lie_alg_basis_mul: [lat_dim, lat_dim, mat_dim, mat_dim]
        lie_act_mul = tf.matmul(lie_alg_basis_mul, act_points)
        # [lat_dim, lat_dim, mat_dim, n_act_points]
        # print('lie_act_mul.shape:', lie_act_mul.get_shape().as_list())
        hessian_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(lie_act_mul), axis=[2, 3]))
    else:
        raise ValueError('Not recognized hessian_type:', hessian_type)
    hessian_loss = autosummary('Loss/lie_vae_hessian_loss', hessian_loss)

    # loss = hy_gmat * gmat_loss + hy_dcp * spl_loss + hy_hes * hessian_loss + hy_lin * lin_loss
    loss = hy_gmat * gmat_loss + hy_hes * hessian_loss + hy_lin * lin_loss
    return loss

def group_act_spl_vae(E,
                      G,
                      opt,
                      training_set,
                      minibatch_size,
                      reals,
                      labels,
                      latent_type='normal',
                      hy_beta=1,
                      hy_dcp=0,
                      hy_gmat=0,
                      hy_hes=0,
                      hy_lin=0,
                      hy_ncut=0,
                      hy_rec=0,
                      hessian_type='no_act_points',
                      recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var, group_feats_E = get_return_v(
        E.get_output_for(reals, labels, is_training=True), 3)
    print('group_feats_E.shape:', group_feats_E.get_shape().as_list())
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

    reconstructions, group_feats_G, _, _, lie_alg_feats, lie_alg_basis, act_points = get_return_v(
        G.get_output_for(sampled_all, labels_all, is_training=True), 7)
    lie_group_loss = make_lie_group_act_spl_loss(group_feats_E=group_feats_E,
                                                 group_feats_G=group_feats_G,
                                                 lie_alg_feats=lie_alg_feats,
                                                 lie_alg_basis=lie_alg_basis,
                                                 act_points=act_points,
                                                 minibatch_size=minibatch_size,
                                                 hy_rec=hy_rec,
                                                 hy_dcp=hy_dcp,
                                                 hy_gmat=hy_gmat,
                                                 hy_hes=hy_hes,
                                                 hy_lin=hy_lin,
                                                 hy_ncut=hy_ncut,
                                                 hessian_type=hessian_type)
    lie_group_loss = autosummary('Loss/lie_group_loss', lie_group_loss)

    reconstruction_loss = make_reconstruction_loss(
        reals, reconstructions[:minibatch_size], recons_type=recons_type)
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)

    elbo = reconstruction_loss + hy_beta * kl_loss
    elbo = autosummary('Loss/lie_vae_elbo', elbo)
    loss = elbo + lie_group_loss

    loss = autosummary('Loss/lie_vae_loss', loss)
    return loss
