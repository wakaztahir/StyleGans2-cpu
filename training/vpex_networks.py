#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vpex_networks.py
# --- Creation Date: 07-09-2020
# --- Last Modified: Mon 14 Sep 2020 15:25:24 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
VPEX networks.
"""
import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict

from training.networks_stylegan2 import apply_bias_act
from training.networks_stylegan2 import dense_layer, conv2d_layer
from training.networks_stylegan2 import minibatch_stddev_layer
from training.networks_stylegan2 import modulated_conv2d_layer
from dnnlib.tflib.ops.upfirdn_2d import downsample_2d

#----------------------------------------------------------------------------

def vpex_net(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        fake2,  # Second input: hidden features from z + delta(z) [minibatch, channel, height, width].
        latents,  # Ground-truth latent code for fake1.
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        D_global_size=0,
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
        connect_mode='concat',  # How fake1 and fake2 connected.
        return_atts=False,  # If return I_atts.
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
    latents.set_shape([None, dlatent_size])
    fake1 = tf.cast(fake1, dtype)
    fake2 = tf.cast(fake2, dtype)
    latents = tf.cast(latents, dtype)
    if connect_mode == 'diff':
        images_in = fake1 - fake2
    elif connect_mode == 'concat':
        images_in = tf.concat([fake1, fake2], axis=1)

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

    # attention features for each latent dimension.
    def get_att_map(latents, x=None):
        with tf.variable_scope('create_att_feats'):
            x_ch, x_h, x_w = x.get_shape().as_list()[1:]
            att_feats = tf.get_variable('att_feats', shape=[1, dlatent_size, x_ch, x_h, x_w],
                                        initializer=tf.initializers.random_normal())
            att_feats = tf.tile(tf.cast(att_feats, dtype), [tf.shape(latents)[0], 1, 1, 1, 1])
            latents = latents[:, tf.newaxis, :]
            latents = tf.tile(latents, [1, dlatent_size, 1])
            latents = tf.reshape(latents, [-1, dlatent_size])
            # att_map = apply_bias_act(modulated_conv2d_layer(att_feats, latents, fmaps=64, kernel=3,
                                                            # demodulate=False, fused_modconv=False),
                                     # act=act) # shape: [b*dlatent_size, 1, 8, 8]
            if x is None:
                att_map = att_feats
                att_map = tf.reshape(att_map, [-1, x_ch, x_h, x_w])
                map_ch = x_ch
            else:
                x = tf.reshape(x, [-1, 1, x_ch, x_h, x_w])
                x = tf.tile(x, [1, dlatent_size, 1, 1, 1])
                att_map = tf.concat([x, att_feats], axis=2)
                att_map = tf.reshape(att_map, [-1, 2 * x_ch, x_h, x_w])
                map_ch = 2 * x_ch
            with tf.variable_scope('att_conv_3x3'):
                att_map = apply_bias_act(conv2d_layer(att_map,
                                                      fmaps=map_ch,
                                                      kernel=3),
                                         act=act)
            with tf.variable_scope('att_conv_1x1'):
                att_map = apply_bias_act(conv2d_layer(att_map, fmaps=1, kernel=1))
            att_map = tf.reshape(att_map, [-1, dlatent_size, 1, x_h*x_w])
            att_map = tf.nn.softmax(att_map, axis=-1)
            # att_map = tf.nn.sigmoid(att_map)
            # att_map = tf.reshape(att_map, [-1, dlatent_size, 1, 8, 8])
        return att_map

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 3, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Duplicate for each att.
    with tf.variable_scope('apply_att'):
        att_map = get_att_map(latents, x)
        x_ch, x_h, x_w = x.get_shape().as_list()[1:]
        assert x_h == 8
        x_ori = tf.reshape(x, [-1, 1, x_ch, x_h * x_w]) # [b, 1, ch, h*w]
        x = tf.reshape(x, [-1, 1, x_ch, x_h * x_w])
        x = att_map * x
        x = tf.reduce_sum(x, axis=-1) # [b, dlatent, ch]
        x = tf.reshape(x, [-1, x_ch, 1, 1]) # [b * dlatent, ch, 1, 1]
        with tf.variable_scope('after_att_conv_1x1'):
            x = apply_bias_act(conv2d_layer(x, fmaps=x_ch, kernel=1))
        x = tf.reshape(x, [-1, dlatent_size, x_ch, 1]) # [b, dlatent, ch, 1]

        x = tf.tile(x, [1, 1, 1, x_h * x_w])
        # x = x + x_ori # [b, dlatent, ch, h * w]
        x = tf.reshape(x, [-1, x_ch, x_h, x_w])
        y_ch, y_h, y_w = y.get_shape().as_list()[1:]
        y = y[:, tf.newaxis, ...]
        y = tf.tile(y, [1, dlatent_size, 1, 1, 1])
        y = tf.reshape(y, [-1, y_ch, y_h, y_w])

    for res in range(3, 2, -1):
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
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    with tf.variable_scope('Output'):
        with tf.variable_scope('Dense_VC'):
            x = apply_bias_act(dense_layer(x, fmaps=1))

    with tf.variable_scope('Final_reshape_x'):
        x = tf.reshape(x, [-1, dlatent_size])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    if return_atts:
        with tf.variable_scope('Reshape_atts'):
            att_map = tf.reshape(att_map, [-1, 8, 8, 1])
            att_map = tf.image.resize(att_map, size=(resolution, resolution))
            att_map = tf.reshape(att_map, [-1, dlatent_size, 1, resolution, resolution])
        return x, att_map
    else:
        return x
