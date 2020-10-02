#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_group_networks_v2.py
# --- Creation Date: 27-09-2020
# --- Last Modified: Fri 02 Oct 2020 03:39:28 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
GroupVAE v2.
"""
import math
import tensorflow as tf
from training.loss_vae import split_latents


def build_group_act_sim_prior_G(latents_in,
                                    name,
                                    scope_idx,
                                    group_feats_size,
                                    n_act_points,
                                    lie_alg_init_type='oth',
                                    lie_alg_init_scale=0.1,
                                    is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        lie_alg_basis_ls = []
        latent_dim = latents_in.get_shape().as_list()[-1]
        mat_dim = int(math.sqrt(group_feats_size))
        for i in range(latent_dim):
            init = tf.initializers.random_normal(0, lie_alg_init_scale)
            lie_alg_tmp = tf.get_variable('lie_alg_' + str(i),
                                          shape=[1, mat_dim, mat_dim],
                                          initializer=init)
            if lie_alg_init_type == 'oth':
                lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
                lie_alg_tmp = lie_alg_tmp - tf.transpose(lie_alg_tmp, perm=[0, 2, 1])
            lie_alg_basis_ls.append(lie_alg_tmp)
        lie_alg_basis = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]

        lie_alg_mul = latents_in[
            ..., tf.newaxis, tf.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
        lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]

        # lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])
        act_init = tf.initializers.random_normal(0, 0.01)
        act_points = tf.get_variable('act_points',
                                     shape=[1, mat_dim, n_act_points],
                                     initializer=act_init)
        # transed_act_points = tf.matmul(lie_group, act_points)
        # transed_act_points_tensor = tf.reshape(transed_act_points,
                                               # [-1, mat_dim * n_act_points])
        transed_act_points_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])

        d1 = tf.layers.dense(transed_act_points_tensor, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, lie_group, lie_alg, lie_alg_basis, act_points


def build_group_act_spl_sim_prior_G(latents_in,
                                    name,
                                    scope_idx,
                                    group_feats_size,
                                    n_act_points,
                                    lie_alg_init_type='oth',
                                    lie_alg_init_scale=0.1,
                                    is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        lie_alg_basis_ls = []
        latent_dim = latents_in.get_shape().as_list()[-1]
        latents_in_cut_ls = split_latents(latents_in, hy_ncut=1) # [x0, x1]

        mat_dim = int(math.sqrt(group_feats_size))
        for i in range(latent_dim):
            init = tf.initializers.random_normal(0, lie_alg_init_scale)
            lie_alg_tmp = tf.get_variable('lie_alg_' + str(i),
                                          shape=[1, mat_dim, mat_dim],
                                          initializer=init)
            if lie_alg_init_type == 'oth':
                lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
                lie_alg_tmp = lie_alg_tmp - tf.transpose(lie_alg_tmp, perm=[0, 2, 1])
            lie_alg_basis_ls.append(lie_alg_tmp)
        lie_alg_basis = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]

        if is_validation:
            lie_alg_mul = latents_in[
                ..., tf.newaxis, tf.
                newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
            lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
            lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
        else:
            lie_alg_mul_0 = latents_in_cut_ls[0][
                ..., tf.newaxis, tf.
                newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
            lie_alg_mul_1 = latents_in_cut_ls[1][
                ..., tf.newaxis, tf.
                newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
            lie_alg_0 = tf.reduce_sum(lie_alg_mul_0, axis=1)  # [b, mat_dim, mat_dim]
            lie_alg_1 = tf.reduce_sum(lie_alg_mul_1, axis=1)  # [b, mat_dim, mat_dim]
            lie_alg = lie_alg_0 + lie_alg_1
            lie_group_0 = tf.linalg.expm(lie_alg_0)  # [b, mat_dim, mat_dim]
            lie_group_1 = tf.linalg.expm(lie_alg_1)  # [b, mat_dim, mat_dim]
            lie_group = tf.matmul(lie_group_0, lie_group_1)

        # lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])
        act_init = tf.initializers.random_normal(0, 0.01)
        act_points = tf.get_variable('act_points',
                                     shape=[1, mat_dim, n_act_points],
                                     initializer=act_init)
        # transed_act_points = tf.matmul(lie_group, act_points)
        # transed_act_points_tensor = tf.reshape(transed_act_points,
                                               # [-1, mat_dim * n_act_points])
        transed_act_points_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])

        d1 = tf.layers.dense(transed_act_points_tensor, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, lie_group, lie_alg, lie_alg_basis, act_points
