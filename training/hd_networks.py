#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: hd_networks.py
# --- Creation Date: 07-04-2020
# --- Last Modified: Sun 12 Apr 2020 17:15:32 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
HD disentanglement networks.
"""

import numpy as np
import tensorflow as tf
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d
from training.networks_stylegan2 import get_weight, dense_layer, conv2d_layer
from training.networks_stylegan2 import apply_bias_act, naive_upsample_2d
from training.networks_stylegan2 import minibatch_stddev_layer

#----------------------------------------------------------------------------
# Mapper disentanglement network.
def net_M(latents_in,
          C_global_size=10,
          D_global_size=0,
          latent_size=512,  # Latent vector (Z) dimensionality.
          mapping_layers=4,  # Number of mapping layers.
          mapping_lrmul=0.1,  # Learning rate multiplier for the mapping layers.
          mapping_fmaps=512,  # Number of activations in the mapping layers.
          mapping_nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
          dtype='float32',  # Data type to use for activations and outputs.
          **_kwargs):  # Ignore unrecognized keyword args.
    act = mapping_nonlinearity

    latents_in.set_shape([None, C_global_size + D_global_size])
    x = latents_in
    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            # if layer_idx == mapping_layers - 1:
                # fmaps = latent_size
                # act = 'tanh'
            # else:
                # fmaps = mapping_fmaps
                # act = mapping_nonlinearity
            # x = apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul),
                               # act=act, lrmul=mapping_lrmul)
            if layer_idx == mapping_layers - 1:
                fmaps = latent_size
                act = 'linear'
            else:
                fmaps = mapping_fmaps
                act = mapping_nonlinearity
            x = apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul),
                               act=act, lrmul=mapping_lrmul)
    # # x = x * 1.5
    # with tf.variable_scope('Dense1'):
        # # x = tf.zeros([tf.shape(x)[0], latent_size], dtype=x.dtype) + 0.5
        # x = tf.random.normal([tf.shape(x)[0], latent_size], mean=0.0, stddev=0.5)

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='to_latent_out')

#----------------------------------------------------------------------------
# Recognizor network.

def net_I(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        fake2,  # Second input: hidden features from z + delta(z) [minibatch, channel, height, width].
        C_global_size=10,
        D_global_size=0,
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        fmap_base=16 << 10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        connect_mode='concat',  # How fake1 and fake2 connected.
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    fake1.set_shape([None, num_channels, resolution, resolution])
    fake2.set_shape([None, num_channels, resolution, resolution])
    fake1 = tf.cast(fake1, dtype)
    fake2 = tf.cast(fake2, dtype)
    if connect_mode == 'diff':
        images_in = fake1 - fake2
    elif connect_mode == 'concat':
        images_in = tf.concat([fake1, fake2], axis=1)

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res - 1), kernel=1), act=act)
            return t if x is None else x + t

    def block(x, res):  # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 2), kernel=3, down=True,
                                            resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res - 2), kernel=1, down=True,
                                 resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    def hier_out_branch(x, nd_out):
        with tf.variable_scope('Output'):
            if len(x.shape) == 4:
                x = tf.reduce_mean(tf.reduce_mean(x, axis=3), axis=2)
            elif len(x.shape) != 2:
                raise ValueError('Not recognized dimension.')
            x = apply_bias_act(dense_layer(x, fmaps=nd_out))
        return x

    # Main layers.
    nd_out_base = C_global_size // (resolution_log2 - 1)
    nd_out_list = [nd_out_base + C_global_size % (resolution_log2 - 1) if i == 0 else nd_out_base for i in range(resolution_log2 - 1)]
    out_list = []
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)
            x_out_branch = hier_out_branch(x, nd_out_list[res-2])
            out_list.append(x_out_branch)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size,
                                           mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)
        x_out_branch = hier_out_branch(x, nd_out_list[0])
        out_list.append(x_out_branch)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    # with tf.variable_scope('Output'):
        # with tf.variable_scope('Dense_VC'):
            # x = apply_bias_act(dense_layer(x, fmaps=(D_global_size + C_global_size)))

    # Output.
    # assert x.dtype == tf.as_dtype(dtype)
    # return x
    assert out_list[-1].dtype == tf.as_dtype(dtype)
    return tuple(out_list)

#----------------------------------------------------------------------------
# Info-Gan Discriminator network.

def net_I_info(
        images_in,  # First input: generated image from z [minibatch, channel, height, width].
        C_global_size=10,
        D_global_size=0,
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        fmap_base=16 << 10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res - 1), kernel=1),
                               act=act)
            return t if x is None else x + t

    def block(x, res):  # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 2), kernel=3, down=True,
                                            resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res - 2), kernel=1, down=True,
                                 resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"

    with tf.variable_scope('Output'):
        with tf.variable_scope('Dense_VC'):
            x = apply_bias_act(dense_layer(x, fmaps=D_global_size))

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x
