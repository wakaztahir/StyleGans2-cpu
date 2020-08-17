#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_standard_networks.py
# --- Creation Date: 14-08-2020
# --- Last Modified: Mon 17 Aug 2020 15:44:49 AEST
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

def build_standard_conv_G_64(d2_reshaped, name, scope_idx, output_shape, is_validation):
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
        if is_validation:
            d6 = tf.nn.sigmoid(d6)
    return d6

def build_standard_conv_G_128(d2_reshaped, name, scope_idx, output_shape, is_validation):
    pass

def build_standard_fc_D_64(latents, name, scope_idx):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d1 = tf.layers.dense(latents, 1000, activation=tf.nn.leaky_relu, name="d1")
        d2 = tf.layers.dense(d1, 1000, activation=tf.nn.leaky_relu, name="d2")
        d3 = tf.layers.dense(d2, 1000, activation=tf.nn.leaky_relu, name="d3")
        d4 = tf.layers.dense(d3, 1000, activation=tf.nn.leaky_relu, name="d4")
        d5 = tf.layers.dense(d4, 1000, activation=tf.nn.leaky_relu, name="d5")
        d6 = tf.layers.dense(d5, 1000, activation=tf.nn.leaky_relu, name="d6")
        logits = tf.layers.dense(d6, 2, activation=None, name="logits")
        probs = tf.nn.softmax(logits)
    return logits, probs

def build_standard_fc_D_128(latents, name, scope_idx):
    pass
