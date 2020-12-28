#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_group_networks_v3.py
# --- Creation Date: 14-12-2020
# --- Last Modified: Sun 27 Dec 2020 21:57:37 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
GroupVAE v3.
"""
import math
import tensorflow as tf
from training.loss_vae import split_latents


def build_group_norm_prior_G(latents_in,
                             name,
                             scope_idx,
                             group_feats_size,
                             hy_ncut=0,
                             lie_alg_init_type='none',
                             lie_alg_init_scale=0.1,
                             normalize_alg=True,
                             use_alg_var=True,
                             is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        lie_alg_basis_norm_ls = []
        lie_var_ls = []
        lie_alg_basis_ls = []
        latent_dim = latents_in.get_shape().as_list()[-1]
        latents_in_cut_ls = split_latents(latents_in,
                                          hy_ncut=hy_ncut)  # [x0, x1, ...]

        mat_dim = int(math.sqrt(group_feats_size))
        assert mat_dim * mat_dim == group_feats_size
        for i in range(latent_dim):
            init = tf.initializers.random_normal(0, lie_alg_init_scale)
            lie_alg_tmp = tf.get_variable('lie_alg_' + str(i),
                                          shape=[1, mat_dim, mat_dim],
                                          initializer=init)
            if lie_alg_init_type == 'oth':
                lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
                lie_alg_tmp = lie_alg_tmp - tf.transpose(lie_alg_tmp,
                                                         perm=[0, 2, 1])
            if normalize_alg:
                lie_alg_tmp_norm, _ = tf.linalg.normalize(lie_alg_tmp,
                                                          axis=[-2, -1])
            else:
                lie_alg_tmp_norm = lie_alg_tmp

            if use_alg_var:
                var_tmp = tf.get_variable(
                    'lie_alg_var_' + str(i), shape=[1, 1],
                    initializer=init)  # lie_alg_whole = var * lie_alg
            else:
                var_tmp = tf.ones([1, 1], dtype=lie_alg_tmp.dtype)
            lie_alg_basis_norm_ls.append(lie_alg_tmp_norm)
            lie_var_ls.append(var_tmp)
            lie_alg_basis_ls.append(lie_alg_tmp_norm * var_tmp)
        lie_vars = tf.concat(lie_var_ls, axis=1)  # [1, lat_dim]
        lie_alg_basis_norm = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_alg_basis = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]

        if is_validation:
            lie_alg_mul = latents_in[
                ..., tf.newaxis, tf.
                newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
            lie_alg = tf.reduce_sum(lie_alg_mul,
                                    axis=1)  # [b, mat_dim, mat_dim]
            lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
        else:
            lie_group = tf.eye(
                mat_dim,
                dtype=lie_alg_basis.dtype)[tf.newaxis,
                                           ...]  # [1, mat_dim, mat_dim]
            lie_alg = 0.
            for masked_latent in latents_in_cut_ls:
                lie_alg_sum_tmp = tf.reduce_sum(
                    masked_latent[..., tf.newaxis, tf.newaxis] * lie_alg_basis,
                    axis=1)
                lie_alg += lie_alg_sum_tmp  # [b, mat_dim, mat_dim]
                lie_group_tmp = tf.linalg.expm(lie_alg_sum_tmp)
                lie_group = tf.matmul(lie_group,
                                      lie_group_tmp)  # [b, mat_dim, mat_dim]

        lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])

        d1 = tf.layers.dense(lie_group_tensor, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, lie_group, lie_alg, lie_alg_basis_norm, lie_vars
