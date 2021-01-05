#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_group_networks_v4.py
# --- Creation Date: 27-12-2020
# --- Last Modified: Wed 06 Jan 2021 00:21:12 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
GroupVAE v4.
"""
import math
import tensorflow as tf
from training.loss_vae import split_latents


def val_exp(x, lie_alg_basis_ls):
    lie_alg_basis = tf.concat(lie_alg_basis_ls,
                              axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]
    lie_alg_mul = x[..., tf.newaxis, tf.
                    newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
    lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
    lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
    return lie_group


def train_exp(x, lie_alg_basis_ls, hy_ncut, mat_dim):
    lie_alg_basis = tf.concat(lie_alg_basis_ls,
                              axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]
    lie_group = tf.eye(mat_dim, dtype=x.dtype)[tf.newaxis,
                                               ...]  # [1, mat_dim, mat_dim]
    lie_alg = 0.
    latents_in_cut_ls = split_latents(x, hy_ncut=hy_ncut)  # [x0, x1, ...]
    for masked_latent in latents_in_cut_ls:
        lie_alg_sum_tmp = tf.reduce_sum(
            masked_latent[..., tf.newaxis, tf.newaxis] * lie_alg_basis, axis=1)
        lie_alg += lie_alg_sum_tmp  # [b, mat_dim, mat_dim]
        lie_group_tmp = tf.linalg.expm(lie_alg_sum_tmp)
        lie_group = tf.matmul(lie_group,
                              lie_group_tmp)  # [b, mat_dim, mat_dim]
    return lie_group


def init_alg_basis(i, j, mat_dim, lie_alg_init_type, lie_alg_init_scale,
                   normalize_alg, use_alg_var):
    init = tf.initializers.random_normal(0, lie_alg_init_scale)
    lie_alg_tmp = tf.get_variable('lie_alg_%d_%d' % (i, j),
                                  shape=[1, mat_dim, mat_dim],
                                  initializer=init)
    if lie_alg_init_type == 'oth':
        lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
        lie_alg_tmp = lie_alg_tmp - tf.transpose(lie_alg_tmp, perm=[0, 2, 1])
    if normalize_alg:
        lie_alg_tmp_norm, _ = tf.linalg.normalize(lie_alg_tmp, axis=[-2, -1])
    else:
        lie_alg_tmp_norm = lie_alg_tmp

    if use_alg_var:
        var_tmp = tf.get_variable(
            'lie_alg_var_%d_%d' % (i, j), shape=[1, 1],
            initializer=init)  # lie_alg_whole = var * lie_alg
    else:
        var_tmp = tf.ones([1, 1], dtype=lie_alg_tmp.dtype)
    return lie_alg_tmp_norm * var_tmp, lie_alg_tmp_norm, var_tmp


def build_group_subspace_post_E(x,
                                name,
                                scope_idx,
                                subgroup_sizes_ls,
                                subspace_sizes_ls,
                                latent_size,
                                is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        flat_x = tf.layers.flatten(x)
        e5 = tf.layers.dense(flat_x, 256, activation=tf.nn.relu, name="e5")

        # Group feats mapping.
        group_feats_size = sum(subgroup_sizes_ls)
        group_feats = tf.layers.dense(e5,
                                      group_feats_size,
                                      activation=None,
                                      name="group_feats_E")
        b_idx = 0
        means_ls = []
        logvar_ls = []
        for i, subgroup_size_i in enumerate(subgroup_sizes_ls):
            e_idx = b_idx + subgroup_size_i
            means_ls.append(
                tf.layers.dense(group_feats[:, b_idx:e_idx],
                                subspace_sizes_ls[i],
                                activation=None,
                                name="means_%d" % i))
            logvar_ls.append(
                tf.layers.dense(group_feats[:, b_idx:e_idx],
                                subspace_sizes_ls[i],
                                activation=None,
                                name="logvar_%d" % i))
            b_idx = e_idx
        means = tf.concat(means_ls, axis=1)
        log_var = tf.concat(logvar_ls, axis=1)
    return means, log_var, group_feats


def build_group_subspace_prior_G(latents_in,
                                 group_feats_E,
                                 name,
                                 scope_idx,
                                 subgroup_sizes_ls,
                                 subspace_sizes_ls,
                                 lie_alg_init_type_ls,
                                 hy_ncut=0,
                                 lie_alg_init_scale=0.001,
                                 normalize_alg=False,
                                 use_alg_var=False,
                                 forward_eg_prob=0.3333,
                                 is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        lie_alg_basis_norm_ls = []
        lie_var_ls = []
        lie_alg_basis_ls = []
        lie_alg_basis_flattened_ls = []
        latent_dim = latents_in.get_shape().as_list()[-1]
        assert latent_dim == sum(subspace_sizes_ls)
        assert len(subgroup_sizes_ls) == len(subspace_sizes_ls)

        # Init lie_alg for each latent dim.
        for i, subgroup_size_i in enumerate(subgroup_sizes_ls):
            mat_dim = int(math.sqrt(subgroup_size_i))
            assert mat_dim * mat_dim == subgroup_size_i
            for j in range(subspace_sizes_ls[i]):
                lie_alg_tmp, _, var_tmp = init_alg_basis(
                    i, j, mat_dim, lie_alg_init_type_ls[i], lie_alg_init_scale,
                    normalize_alg, use_alg_var)
                lie_alg_basis_ls.append(lie_alg_tmp)
                lie_alg_basis_flattened_ls.append(
                    tf.reshape(lie_alg_tmp, [1, -1]))
                # lie_alg_basis_norm_ls.append(lie_alg_tmp_norm)
                lie_var_ls.append(var_tmp)
        lie_vars = tf.concat(lie_var_ls, axis=1)  # [1, lat_dim]
        # lie_alg_basis_norm = tf.concat(
        # lie_alg_basis_ls, axis=0)[tf.newaxis,
        # ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_alg_basis = tf.concat(
            lie_alg_basis_flattened_ls,
            axis=1)  # [1, mat_dim_1*mat_dim_1+mat_dim_2*mat_dim_2+...]

        # Calc exp.
        lie_group_tensor_ls = []
        b_idx = 0
        for i, subgroup_size_i in enumerate(subgroup_sizes_ls):
            mat_dim = int(math.sqrt(subgroup_size_i))
            e_idx = b_idx + subspace_sizes_ls[i]
            if subspace_sizes_ls[i] > 1:
                if is_validation:
                    lie_subgroup = val_exp(latents_in[:, b_idx:e_idx],
                                           lie_alg_basis_ls[b_idx:e_idx])
                else:
                    lie_subgroup = train_exp(latents_in[:, b_idx:e_idx],
                                             lie_alg_basis_ls[b_idx:e_idx],
                                             hy_ncut, mat_dim)
            else:
                lie_subgroup = val_exp(latents_in[:, b_idx:e_idx],
                                       lie_alg_basis_ls[b_idx:e_idx])
            lie_subgroup_tensor = tf.reshape(lie_subgroup,
                                             [-1, mat_dim * mat_dim])
            lie_group_tensor_ls.append(lie_subgroup_tensor)
            b_idx = e_idx

        if is_validation:
            lie_group_tensor = tf.concat(
                    lie_group_tensor_ls, axis=1)  # [b, group_feat_size]
        else:
            eg_prob = tf.random.uniform(shape=[])
            lie_group_tensor = tf.cond(
                eg_prob < forward_eg_prob, lambda: group_feats_E, lambda: tf.concat(
                    lie_group_tensor_ls, axis=1))  # [b, group_feat_size]

        d1 = tf.layers.dense(lie_group_tensor, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, lie_group_tensor, lie_vars, lie_alg_basis, lie_vars
