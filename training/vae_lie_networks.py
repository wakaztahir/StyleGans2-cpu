#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_lie_networks.py
# --- Creation Date: 21-09-2020
# --- Last Modified: Sun 27 Sep 2020 02:52:20 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie VAE networks.
"""
import math
import tensorflow as tf


def build_lie_sim_prior_G(latents_in,
                          name,
                          scope_idx,
                          group_feats_size,
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
            lie_alg_basis_ls.append(lie_alg_tmp)
        lie_alg_basis = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]
        # lie_alg_basis = tf.reshape(lie_alg_basis, [1, latent_dim, mat_dim * mat_dim])
        # lie_alg_basis = tf.math.l2_normalize(lie_alg_basis, axis=-1)
        # lie_alg_basis = tf.reshape(lie_alg_basis, [1, latent_dim, mat_dim, mat_dim])

        lie_alg_mul = latents_in[
            ..., tf.newaxis, tf.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
        lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
        lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])
        # group_feats = tf.layers.dense(latents_in,
        # group_feats_size,
        # activation=None)

        d1 = tf.layers.dense(lie_group_tensor, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, lie_group, lie_alg, lie_alg_basis

def build_lie_sim_prior_G_oth(latents_in,
                          name,
                          scope_idx,
                          group_feats_size,
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
            lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
            lie_alg_tmp = lie_alg_tmp - tf.transpose(lie_alg_tmp, perm=[0, 2, 1])
            lie_alg_basis_ls.append(lie_alg_tmp)
        lie_alg_basis = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]
        # lie_alg_basis = tf.reshape(lie_alg_basis, [1, latent_dim, mat_dim * mat_dim])
        # lie_alg_basis = tf.math.l2_normalize(lie_alg_basis, axis=-1)
        # lie_alg_basis = tf.reshape(lie_alg_basis, [1, latent_dim, mat_dim, mat_dim])

        lie_alg_mul = latents_in[
            ..., tf.newaxis, tf.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
        lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
        lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])
        # group_feats = tf.layers.dense(latents_in,
        # group_feats_size,
        # activation=None)

        d1 = tf.layers.dense(lie_group_tensor, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, lie_group, lie_alg, lie_alg_basis

def build_lie_sim_prior_G_oth_l2(latents_in,
                                 name,
                                 scope_idx,
                                 group_feats_size,
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
            lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
            lie_alg_tmp = lie_alg_tmp - tf.transpose(lie_alg_tmp, perm=[0, 2, 1])
            lie_alg_basis_ls.append(lie_alg_tmp)
        lie_alg_basis = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_alg_basis = tf.reshape(lie_alg_basis, [1, latent_dim, mat_dim * mat_dim])
        lie_alg_basis = tf.math.l2_normalize(lie_alg_basis, axis=-1)
        lie_alg_basis = tf.reshape(lie_alg_basis, [1, latent_dim, mat_dim, mat_dim])

        lie_alg_mul = latents_in[
            ..., tf.newaxis, tf.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
        lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
        lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])
        # group_feats = tf.layers.dense(latents_in,
        # group_feats_size,
        # activation=None)

        d1 = tf.layers.dense(lie_group_tensor, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, lie_group, lie_alg, lie_alg_basis

def build_lie_sim_prior_G_oth_squash(latents_in,
                                     name,
                                     scope_idx,
                                     group_feats_size,
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
            lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
            lie_alg_tmp = lie_alg_tmp - tf.transpose(lie_alg_tmp, perm=[0, 2, 1])
            lie_alg_basis_ls.append(lie_alg_tmp)
        lie_alg_basis = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_alg_basis = tf.reshape(lie_alg_basis, [1, latent_dim, mat_dim * mat_dim])
        lie_alg_basis_norm = tf.norm(lie_alg_basis, axis=-1, keepdims=True)
        lie_alg_basis_norm_square = tf.square(lie_alg_basis_norm)
        lie_alg_basis_squash = lie_alg_basis_norm_square / (1 + lie_alg_basis_norm_square) * \
            lie_alg_basis / lie_alg_basis_norm
        lie_alg_basis = tf.reshape(lie_alg_basis_squash, [1, latent_dim, mat_dim, mat_dim])

        lie_alg_mul = latents_in[
            ..., tf.newaxis, tf.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
        lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
        lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])
        # group_feats = tf.layers.dense(latents_in,
        # group_feats_size,
        # activation=None)

        d1 = tf.layers.dense(lie_group_tensor, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, lie_group, lie_alg, lie_alg_basis

def build_lie_sim_prior_G_oth_nogroup(latents_in,
                                      name,
                                      scope_idx,
                                      group_feats_size,
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
            lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
            lie_alg_tmp = lie_alg_tmp - tf.transpose(lie_alg_tmp, perm=[0, 2, 1])
            lie_alg_basis_ls.append(lie_alg_tmp)
        lie_alg_basis = tf.concat(
            lie_alg_basis_ls, axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]
        # lie_alg_basis = tf.reshape(lie_alg_basis, [1, latent_dim, mat_dim * mat_dim])
        # lie_alg_basis = tf.math.l2_normalize(lie_alg_basis, axis=-1)
        # lie_alg_basis = tf.reshape(lie_alg_basis, [1, latent_dim, mat_dim, mat_dim])

        lie_alg_mul = latents_in[
            ..., tf.newaxis, tf.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
        # lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
        lie_group = lie_alg
        lie_group_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])
        # group_feats = tf.layers.dense(latents_in,
        # group_feats_size,
        # activation=None)

        d1 = tf.layers.dense(lie_group_tensor, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, lie_group, lie_alg, lie_alg_basis
