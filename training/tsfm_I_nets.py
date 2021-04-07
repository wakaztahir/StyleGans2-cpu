#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: tsfm_I_nets.py
# --- Creation Date: 05-04-2021
# --- Last Modified: Mon 05 Apr 2021 22:49:42 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Transformer I Networks.
"""

import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
from dnnlib.tflib.ops.upfirdn_2d import downsample_2d
from training.networks_stylegan2 import dense_layer, conv2d_layer
from training.networks_stylegan2 import apply_bias_act
from training.networks_stylegan2 import minibatch_stddev_layer

#----------------------------------------------------------------------------
# Head network of infogan2.

def head_infogan2(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
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

    fake1.set_shape([None, num_channels, resolution, resolution])
    fake1 = tf.cast(fake1, dtype)
    images_in = fake1

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res - 1), kernel=1),
                               act=act)
            return t if x is None else x + t

    def block(x, res):  # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3),
                               act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=nf(res - 2),
                                            kernel=3,
                                            down=True,
                                            resample_kernel=resample_kernel),
                               act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t,
                                 fmaps=nf(res - 2),
                                 kernel=1,
                                 down=True,
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
                x = minibatch_stddev_layer(x, mbstd_group_size,
                                           mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        with tf.variable_scope('Dense_VC'):
            x = apply_bias_act(dense_layer(x, fmaps=dlatent_size))

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x

#----------------------------------------------------------------------------
# Head network of PS-SC.


def head_ps_sc(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
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

    fake1.set_shape([None, num_channels, resolution, resolution])
    fake1 = tf.cast(fake1, dtype)
    images_in = fake1

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res - 1), kernel=1),
                               act=act)
            return t if x is None else x + t

    def block(x, res):  # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3),
                               act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=nf(res - 2),
                                            kernel=3,
                                            down=True,
                                            resample_kernel=resample_kernel),
                               act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t,
                                 fmaps=nf(res - 2),
                                 kernel=1,
                                 down=True,
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
                x = minibatch_stddev_layer(x, mbstd_group_size,
                                           mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        with tf.variable_scope('Dense_VC'):
            x = apply_bias_act(dense_layer(x, fmaps=dlatent_size))

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x
