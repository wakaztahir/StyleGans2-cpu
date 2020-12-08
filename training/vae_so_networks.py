#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_so_networks.py
# --- Creation Date: 05-12-2020
# --- Last Modified: Tue 08 Dec 2020 18:20:22 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
SO(n) networks.
"""
import math
import numpy as np
import tensorflow as tf


def construct_oneparam_skew_mat(var, mat_dim, var_idx):
    idx_sum = 0
    for i_k, k in enumerate(range(mat_dim - 1, -1, -1)):
        if idx_sum + k > var_idx:
            break
        else:
            idx_sum += k
    j = mat_dim - k + (var_idx - idx_sum)
    i = mat_dim * i_k + j
    skew_mat_1 = tf.reshape(tf.one_hot(i, mat_dim * mat_dim),
                            [1, mat_dim, mat_dim])
    skew_mat = skew_mat_1 - tf.transpose(skew_mat_1, perm=[0, 2, 1])
    oneparam_skew_mat = skew_mat * var
    return oneparam_skew_mat


def get_R_view(mat_dim, lie_alg_init_scale, R_view_scale):
    init_view = tf.initializers.random_normal(0, lie_alg_init_scale)
    R_view_alg = R_view_scale * tf.get_variable(
        'R_view_alg_var', shape=[1, mat_dim, mat_dim], initializer=init_view)
    R_view_alg = tf.matrix_band_part(R_view_alg, 0, -1)
    R_view_alg = R_view_alg - tf.transpose(R_view_alg, perm=[0, 2, 1])
    R_view = tf.linalg.expm(R_view_alg)[tf.newaxis,
                                        ...]  # [1, 1, mat_dim, mat_dim]
    return R_view


def sample_sphere_points(mat_dim, n_points):
    # Return: [1, 1, mat_dim, n_points]
    init = tf.initializers.random_normal(0, 1)
    raw_points = tf.get_variable('sphere_raw_points',
                                 shape=[1, 1, mat_dim, n_points],
                                 trainable=False,
                                 initializer=init)
    points, _ = tf.linalg.normalize(raw_points, axis=2)
    return points


def build_so_prior_G(latents_in,
                     name,
                     scope_idx,
                     group_feats_size,
                     lie_alg_init_scale=0.1,
                     R_view_scale=1,
                     mapping_after_exp=False,
                     use_sphere_points=False,
                     n_sphere_points=100,
                     is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        latent_dim = latents_in.get_shape().as_list()[-1]
        mat_dim = int(math.sqrt(group_feats_size))
        assert mat_dim * (
            mat_dim -
            1) == 2 * latent_dim  # Orthogonal group free degrees check.
        lie_alg_basis_ls = []
        lie_var_ls = []
        for i in range(latent_dim):
            init = tf.initializers.random_normal(0, lie_alg_init_scale)
            var_tmp = tf.get_variable('lie_alg_var_' + str(i),
                                      shape=[1, 1],
                                      initializer=init)
            lie_alg_tmp = construct_oneparam_skew_mat(
                var_tmp, mat_dim, i)  # [1, mat_dim, mat_dim]
            lie_var_ls.append(var_tmp)
            lie_alg_basis_ls.append(lie_alg_tmp)
        lie_vars = tf.concat(lie_var_ls, axis=1)  # [1, lat_dim]
        lie_alg_basis = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]

        lie_algs = latents_in[
            ..., tf.newaxis, tf.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_algs_rsb = tf.reshape(
            lie_algs, [-1, mat_dim, mat_dim])  # [b*lat_dim, mat_dim, mat_dim]
        lie_groups_rsb = tf.linalg.expm(
            lie_algs_rsb)  # [b*lat_dim, mat_dim, mat_dim]
        lie_groups = tf.reshape(lie_groups_rsb,
                                [-1, latent_dim, mat_dim, mat_dim
                                 ])  # [b, lat_dim, mat_dim, mat_dim]

        lie_groups_ls = tf.split(lie_groups, latent_dim, axis=1)
        Rs_ls = []
        R_overall = get_R_view(mat_dim, lie_alg_init_scale,
                               R_view_scale)  # [1, 1, mat_dim, mat_dim]
        if use_sphere_points:
            print('using sphere points')
            sphere_points = sample_sphere_points(
                mat_dim, n_sphere_points)  # [1, 1, mat_dim, n_points]
        for i, R_i in enumerate(lie_groups_ls):
            R_overall = tf.matmul(R_overall, R_i)
            if use_sphere_points:
                sphere_points_rot = tf.matmul(R_overall, sphere_points)
            else:
                sphere_points_rot = R_overall
            Rs_ls.append(sphere_points_rot)

        lie_groups_as_fm = tf.concat(Rs_ls, axis=1)
        # lie_groups_as_tensor = tf.reshape(lie_groups_as_fm,
        # [-1, latent_dim * mat_dim * mat_dim])
        lie_groups_as_tensor = tf.layers.flatten(lie_groups_as_fm)
        if mapping_after_exp:
            feats_0 = tf.layers.dense(lie_groups_as_tensor,
                                      256,
                                      activation=tf.nn.relu)
        else:
            feats_0 = lie_groups_as_tensor
        feats_1 = tf.layers.dense(feats_0, 1024, activation=tf.nn.relu)
        fm = tf.reshape(feats_1, [-1, 64, 4, 4])
    return fm, lie_groups_as_fm, lie_algs, lie_alg_basis, lie_vars
