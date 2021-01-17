#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vae_group_v4.py
# --- Creation Date: 28-12-2020
# --- Last Modified: Sun 17 Jan 2021 16:27:05 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss GroupVAE v4.
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


def calc_basis_mul_ij(lie_alg_basis_ls):
    lie_alg_basis = tf.concat(lie_alg_basis_ls,
                              axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]
    _, lat_dim, mat_dim, _ = lie_alg_basis.get_shape().as_list()
    lie_alg_basis_col = tf.reshape(lie_alg_basis,
                                   [lat_dim, 1, mat_dim, mat_dim])
    lie_alg_basis_outer_mul = tf.matmul(
        lie_alg_basis,
        lie_alg_basis_col)  # [lat_dim, lat_dim, mat_dim, mat_dim]
    hessian_mask = 1. - tf.eye(
        lat_dim,
        dtype=lie_alg_basis_outer_mul.dtype)[:, :, tf.newaxis, tf.newaxis]
    lie_alg_basis_mul_ij = lie_alg_basis_outer_mul * hessian_mask  # XY
    return lie_alg_basis_mul_ij


def calc_hessian_loss(lie_alg_basis_mul_ij, i):
    hessian_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(lie_alg_basis_mul_ij), axis=[2, 3]))
    hessian_loss = autosummary('Loss/hessian_%d' % i, hessian_loss)
    return hessian_loss


def calc_commute_loss(lie_alg_basis_mul_ij, i):
    lie_alg_commutator = lie_alg_basis_mul_ij - tf.transpose(
        lie_alg_basis_mul_ij, [0, 1, 3, 2])
    commute_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(lie_alg_commutator), axis=[2, 3]))
    commute_loss = autosummary('Loss/commute_%d' % i, commute_loss)
    return commute_loss


def basis_flattened_to_ls(lie_alg_basis_flattened, subgroup_sizes_ls,
                          subspace_sizes_ls):
    '''
    lie_alg_basis_flattened: [1, mat_dim_1*mat_dim_1 + mat_dim_2*mat_dim_2 + ...]
    '''
    # lie_alg_basis_ls = []
    # b_idx = 0
    # for i, subgroup_size_i in enumerate(subgroup_sizes_ls):
    # mat_dim = int(math.sqrt(subgroup_size_i))
    # assert mat_dim * mat_dim == subgroup_size_i
    # for j in range(subspace_sizes_ls[i]):
    # e_idx = b_idx + subgroup_size_i
    # lie_alg_basis_ls.append(
    # tf.reshape(lie_alg_basis_flattened[:, b_idx:e_idx],
    # [1, mat_dim, mat_dim]))
    # b_idx = e_idx
    split_ls = []
    for i, subgroup_size_i in enumerate(subgroup_sizes_ls):
        split_ls += [subgroup_size_i] * subspace_sizes_ls[i]
    lie_alg_basis_ls = list(tf.split(lie_alg_basis_flattened, split_ls,
                                     axis=1))
    lie_alg_basis_ls_new = []
    for i, lie_alg_basis_i in enumerate(lie_alg_basis_ls):
        mat_dim = int(math.sqrt(lie_alg_basis_i.get_shape().as_list()[-1]))
        lie_alg_basis_ls_new.append(tf.reshape(lie_alg_basis_i, [1, mat_dim, mat_dim]))
    return lie_alg_basis_ls_new


def make_group_subspace_loss(minibatch_size, group_feats_E, group_feats_G,
                             subgroup_sizes_ls, subspace_sizes_ls,
                             lie_alg_basis_flattened, hy_hes, hy_rec,
                             hy_commute):
    lie_alg_basis_ls = basis_flattened_to_ls(lie_alg_basis_flattened,
                                             subgroup_sizes_ls,
                                             subspace_sizes_ls)
    print('lie_alg_basis_ls.len:', len(lie_alg_basis_ls))
    b_idx = 0
    hessian_loss = 0.
    commute_loss = 0.
    for i, subspace_size in enumerate(subspace_sizes_ls):
        e_idx = b_idx + subspace_size
        if subspace_size > 1:
            mat_dim = int(math.sqrt(subgroup_sizes_ls[i]))
            print('lie_alg_basis_ls[b_idx]:', lie_alg_basis_ls[b_idx])
            assert lie_alg_basis_ls[b_idx].get_shape().as_list()[-1] == mat_dim
            lie_alg_basis_mul_ij = calc_basis_mul_ij(
                lie_alg_basis_ls[b_idx:e_idx])  # XY
            hessian_loss += calc_hessian_loss(lie_alg_basis_mul_ij, i)
            commute_loss += calc_commute_loss(lie_alg_basis_mul_ij, i)
        b_idx = e_idx
    hessian_loss *= hy_hes
    commute_loss *= hy_commute
    rec_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(group_feats_E - group_feats_G), axis=[1]))
    rec_loss = autosummary('Loss/group_feats_rec', rec_loss)
    rec_loss *= hy_rec
    loss = hessian_loss + commute_loss + rec_loss
    return loss


def group_subspace_vae(E,
                       G,
                       opt,
                       training_set,
                       minibatch_size,
                       reals,
                       labels,
                       subgroup_sizes_ls,
                       subspace_sizes_ls,
                       latent_type='normal',
                       hy_beta=1,
                       hy_hes=0,
                       hy_rec=1,
                       hy_commute=0,
                       forward_eg=False,
                       recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var, group_feats_E = get_return_v(
        E.get_output_for(reals, labels, is_training=True), 3)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)

    sampled = sample_from_latent_distribution(means, log_var)

    reconstructions, group_feats_G, _, _, _, lie_alg_basis_flattened, _, _ = get_return_v(
        G.get_output_for(tf.concat([sampled, group_feats_E], axis=1) if forward_eg else sampled,
                         labels,
                         is_training=True), 8)
    lie_group_loss = make_group_subspace_loss(
        minibatch_size=minibatch_size,
        group_feats_E=group_feats_E,
        group_feats_G=group_feats_G,
        subgroup_sizes_ls=subgroup_sizes_ls,
        subspace_sizes_ls=subspace_sizes_ls,
        lie_alg_basis_flattened=lie_alg_basis_flattened,
        hy_hes=hy_hes,
        hy_rec=hy_rec,
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
