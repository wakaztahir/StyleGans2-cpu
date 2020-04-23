#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: hd_networks_stylegan2.py
# --- Creation Date: 22-04-2020
# --- Last Modified: Thu 23 Apr 2020 18:22:38 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
StyleGAN2-like networks used in HD models.
"""

import numpy as np
import tensorflow as tf
# from training.networks_stylegan2 import dense_layer
# from training.networks_stylegan2 import apply_bias_act
import training.networks_stylegan2 as networks_stylegan2

#----------------------------------------------------------------------------
# G_mapping with hd_dis network.
def G_mapping_hd_dis_to_dlatent(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layers          = 8,            # Number of mapping layers.
    mapping_fmaps           = 512,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.

    act = mapping_nonlinearity

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Embed labels and concatenate them with latents.
    if label_size:
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # # Mapping layers.
    # for layer_idx in range(mapping_layers):
        # with tf.variable_scope('Dense%d' % layer_idx):
            # fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            # x = networks_stylegan2.apply_bias_act(networks_stylegan2.dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul), act=act, lrmul=mapping_lrmul)


    x_list = []
    print('dlatent_broadcast:', dlatent_broadcast)
    seg = latent_size // dlatent_broadcast
    remain = latent_size % dlatent_broadcast
    for syn_layer_idx in range(dlatent_broadcast):
        with tf.variable_scope('Dis_to_dlatent%d' % syn_layer_idx):
            if syn_layer_idx == 0:
                start_tmp = 0
                end_tmp = seg + remain
            else:
                start_tmp = end_tmp
                end_tmp = end_tmp + seg
            with tf.variable_scope('Bottleneck'):
                tmp = networks_stylegan2.apply_bias_act(networks_stylegan2.dense_layer(x[:, start_tmp:end_tmp], fmaps=mapping_fmaps, lrmul=mapping_lrmul),
                                     act=act, lrmul=mapping_lrmul)
            with tf.variable_scope('to_dlatent'):
                x_list.append(networks_stylegan2.apply_bias_act(networks_stylegan2.dense_layer(tmp, fmaps=dlatent_size, lrmul=mapping_lrmul),
                                             act=act, lrmul=mapping_lrmul))
    x = tf.stack(x_list, axis=1)

    # # Broadcast.
    # if dlatent_broadcast is not None:
        # with tf.variable_scope('Broadcast'):
            # x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')
