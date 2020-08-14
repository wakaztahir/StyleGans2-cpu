#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_standard_networks.py
# --- Creation Date: 14-08-2020
# --- Last Modified: Sat 15 Aug 2020 02:27:09 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
VAE standard networks.
"""
import tensorflow as tf


def build_standard_conv_E_64(reals_in, name, scope_idx):
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


def build_standard_conv_E_128(reals_in, name, scope_idx):
    pass

def build_standard_conv_G_64(d2_reshaped, name, scope_idx, output_shape):
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
            filters=output_shape[2],
            kernel_size=4,
            strides=2,
            padding="same",
            data_format='channels_first',
        )
    return d6

def build_standard_conv_G_128(d2_reshaped, name, scope_idx, output_shape):
    pass
