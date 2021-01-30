#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_standard_networks.py
# --- Creation Date: 14-08-2020
# --- Last Modified: Sun 31 Jan 2021 02:20:18 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
VAE standard networks.
"""
import tensorflow as tf
from training.utils import get_return_v
from training.vc_modular_networks2 import build_C_spgroup_layers


def build_standard_conv_E_32(reals_in, name, scope_idx, is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        e1 = tf.layers.conv2d(
            inputs=reals_in,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e1",
        )
        e2 = tf.layers.conv2d(
            inputs=e1,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e2",
        )
        e3 = tf.layers.conv2d(
            inputs=e2,
            filters=64,
            kernel_size=2,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e3",
        )
    return e3

def build_standard_conv_E_64(reals_in, name, scope_idx, is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        e1 = tf.layers.conv2d(
            inputs=reals_in,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e1",
        )
        e2 = tf.layers.conv2d(
            inputs=e1,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e2",
        )
        e3 = tf.layers.conv2d(
            inputs=e2,
            filters=64,
            kernel_size=2,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e3",
        )
        e4 = tf.layers.conv2d(
            inputs=e3,
            filters=64,
            kernel_size=2,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e4",
        )
    return e4


def build_standard_conv_E_128(reals_in, name, scope_idx, is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        e1 = tf.layers.conv2d(
            inputs=reals_in,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e1",
        )
        e2 = tf.layers.conv2d(
            inputs=e1,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e2",
        )
        e3 = tf.layers.conv2d(
            inputs=e2,
            filters=64,
            kernel_size=2,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e3",
        )
        e4 = tf.layers.conv2d(
            inputs=e3,
            filters=64,
            kernel_size=2,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e4",
        )
        e5 = tf.layers.conv2d(
            inputs=e4,
            filters=64,
            kernel_size=2,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
            name="e5",
        )
    return e5


def build_standard_post_E(x,
                          name,
                          scope_idx,
                          latent_size,
                          use_relu=True,
                          is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        flat_x = tf.layers.flatten(x)
        e5 = tf.layers.dense(flat_x,
                             256,
                             activation=tf.nn.relu if use_relu else None,
                             name="e5")
        means = tf.layers.dense(e5, latent_size, activation=None, name="means")
        log_var = tf.layers.dense(e5,
                                  latent_size,
                                  activation=None,
                                  name="log_var")
    return means, log_var, e5


def build_standard_prior_G(latents_in,
                           name,
                           scope_idx,
                           use_relu=True,
                           is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d1 = tf.layers.dense(latents_in,
                             256,
                             activation=tf.nn.relu if use_relu else None)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, d1


def build_standard_conv_G_32(d2_reshaped,
                             name,
                             scope_idx,
                             output_shape,
                             recons_type='bernoulli_loss',
                             is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d4 = tf.layers.conv2d_transpose(
            inputs=d2_reshaped,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        d5 = tf.layers.conv2d_transpose(
            inputs=d4,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        d6 = tf.layers.conv2d_transpose(
            inputs=d5,
            filters=output_shape[0],
            kernel_size=4,
            strides=2,
            padding="same",
            data_format='channels_first',
        )
        if is_validation and recons_type == 'bernoulli_loss':
            d6 = tf.nn.sigmoid(d6)
    return d6


def build_standard_conv_G_64(d2_reshaped,
                             name,
                             scope_idx,
                             output_shape,
                             recons_type='bernoulli_loss',
                             is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d3 = tf.layers.conv2d_transpose(
            inputs=d2_reshaped,
            filters=64,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        d4 = tf.layers.conv2d_transpose(
            inputs=d3,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        d5 = tf.layers.conv2d_transpose(
            inputs=d4,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        d6 = tf.layers.conv2d_transpose(
            inputs=d5,
            filters=output_shape[0],
            kernel_size=4,
            strides=2,
            padding="same",
            data_format='channels_first',
        )
        if is_validation and recons_type == 'bernoulli_loss':
            d6 = tf.nn.sigmoid(d6)
    return d6


def build_6layer_conv_G_64(d2_reshaped,
                           name,
                           scope_idx,
                           output_shape,
                           recons_type='bernoulli_loss',
                           is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d3 = tf.layers.conv2d_transpose(
            inputs=d2_reshaped,
            filters=64,
            kernel_size=4,
            strides=1,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )
        d4 = tf.layers.conv2d_transpose(
            inputs=d3,
            filters=64,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )
        d5 = tf.layers.conv2d_transpose(
            inputs=d4,
            filters=32,
            kernel_size=4,
            strides=1,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )
        d6 = tf.layers.conv2d_transpose(
            inputs=d5,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )
        d7 = tf.layers.conv2d_transpose(
            inputs=d6,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )
        d8 = tf.layers.conv2d_transpose(
            inputs=d7,
            filters=output_shape[0],
            kernel_size=4,
            strides=2,
            padding="same",
            data_format='channels_first',
        )
        if is_validation and recons_type == 'bernoulli_loss':
            d8 = tf.nn.sigmoid(d8)
    return d8


def build_standard_conv_G_128(d1_reshaped,
                              name,
                              scope_idx,
                              output_shape,
                              recons_type='bernoulli_loss',
                              is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d2 = tf.layers.conv2d_transpose(
            inputs=d1_reshaped,
            filters=64,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        d3 = tf.layers.conv2d_transpose(
            inputs=d2,
            filters=64,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        d4 = tf.layers.conv2d_transpose(
            inputs=d3,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        d5 = tf.layers.conv2d_transpose(
            inputs=d4,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        d6 = tf.layers.conv2d_transpose(
            inputs=d5,
            filters=output_shape[0],
            kernel_size=4,
            strides=2,
            padding="same",
            data_format='channels_first',
        )
        if is_validation and recons_type == 'bernoulli_loss':
            d6 = tf.nn.sigmoid(d6)
    return d6


def build_fain_conv_G_64(latents_in,
                         name,
                         scope_idx,
                         output_shape,
                         recons_type='bernoulli_loss',
                         is_validation=False):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        latent_size = latents_in.shape[1]
        with tf.variable_scope('4x4Const'):
            x = tf.get_variable('const',
                                shape=[1, 64, 4, 4],
                                initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, latents_in.dtype),
                        [tf.shape(latents_in)[0], 1, 1, 1])

        with tf.variable_scope('FAIN1'):
            # print('x.shape:', x.get_shape().as_list())
            x, atts = get_return_v(
                build_C_spgroup_layers(x,
                                       'SP_latents',
                                       latent_size // 3,
                                       0,
                                       1,
                                       latents_in,
                                       None,
                                       None,
                                       return_atts=True,
                                       resolution=output_shape[1],
                                       n_subs=4), 2)

        x = tf.layers.conv2d_transpose(
            inputs=x,
            filters=64,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        with tf.variable_scope('FAIN2'):
            x, atts = get_return_v(
                build_C_spgroup_layers(x,
                                       'SP_latents',
                                       latent_size // 3,
                                       latent_size // 3,
                                       2,
                                       latents_in,
                                       None,
                                       None,
                                       return_atts=True,
                                       resolution=output_shape[1],
                                       n_subs=4), 2)

        x = tf.layers.conv2d_transpose(
            inputs=x,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        with tf.variable_scope('FAIN3'):
            x, atts = get_return_v(
                build_C_spgroup_layers(x,
                                       'SP_latents',
                                       latent_size - latent_size // 3 * 2,
                                       latent_size // 3 * 2,
                                       3,
                                       latents_in,
                                       None,
                                       None,
                                       return_atts=True,
                                       resolution=output_shape[1],
                                       n_subs=4), 2)

        x = tf.layers.conv2d_transpose(
            inputs=x,
            filters=32,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            padding="same",
            data_format='channels_first',
        )

        x = tf.layers.conv2d_transpose(
            inputs=x,
            filters=output_shape[0],
            kernel_size=4,
            strides=2,
            padding="same",
            data_format='channels_first',
        )
        if is_validation and recons_type == 'bernoulli_loss':
            x = tf.nn.sigmoid(x)
    return x


def build_standard_fc_D_64(latents, name, scope_idx):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d1 = tf.layers.dense(latents,
                             1000,
                             activation=tf.nn.leaky_relu,
                             name="d1")
        d2 = tf.layers.dense(d1, 1000, activation=tf.nn.leaky_relu, name="d2")
        d3 = tf.layers.dense(d2, 1000, activation=tf.nn.leaky_relu, name="d3")
        d4 = tf.layers.dense(d3, 1000, activation=tf.nn.leaky_relu, name="d4")
        d5 = tf.layers.dense(d4, 1000, activation=tf.nn.leaky_relu, name="d5")
        d6 = tf.layers.dense(d5, 1000, activation=tf.nn.leaky_relu, name="d6")
        logits = tf.layers.dense(d6, 2, activation=None, name="logits")
        probs = tf.nn.softmax(logits)
    return logits, probs


def build_standard_fc_sindis_D_64(latents, name, scope_idx):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d1 = tf.layers.dense(latents,
                             1000,
                             activation=tf.nn.leaky_relu,
                             name="d1")
        d2 = tf.layers.dense(d1, 1000, activation=tf.nn.leaky_relu, name="d2")
        d3 = tf.layers.dense(d2, 1000, activation=tf.nn.leaky_relu, name="d3")
        d4 = tf.layers.dense(d3, 1000, activation=tf.nn.leaky_relu, name="d4")
        d5 = tf.layers.dense(d4, 1000, activation=tf.nn.leaky_relu, name="d5")
        d6 = tf.layers.dense(d5, 1000, activation=tf.nn.leaky_relu, name="d6")
        logits = tf.layers.dense(d6, 1, activation=None, name="logits")
    return logits


def build_simple_fc_sindis_D_64(latents, name, scope_idx):
    # with tf.variable_scope(name + '-' + str(scope_idx),
                           # initializer=tf.compat.v1.initializers.he_uniform()):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d1 = tf.layers.dense(latents,
                             256,
                             activation=tf.nn.leaky_relu,
                             name="d1")
        d2 = tf.layers.dense(d1, 128, activation=tf.nn.leaky_relu, name="d2")
        d3 = tf.layers.dense(d2, 64, activation=tf.nn.leaky_relu, name="d3")
        logits = tf.layers.dense(d3, 1, activation=None, name="logits")
    return logits


def build_standard_fc_D_128(latents, name, scope_idx):
    pass
