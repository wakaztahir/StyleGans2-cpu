#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_networks.py
# --- Creation Date: 14-08-2020
# --- Last Modified: Fri 14 Aug 2020 22:23:50 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
VAE networks.
"""
import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
from training.vc_modular_networks2 import split_module_names
from training.vae_standard_networks import build_standard_conv_E_64
from training.vae_standard_networks import build_standard_conv_E_128

#----------------------------------------------------------------------------
# Variation Consistenecy main Generator
def E_main_modular(
        reals_in,  # First input: Real images [minibatch, image_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        input_shape=None,  # Input image shape.
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        dtype='float32',  # Data type to use for activations and outputs.
        fmap_min=16,
        fmap_max=512,
        fmap_decay=0.15,
        latent_size=10,
        module_E_list=None,
        nf_scale=1,
        fmap_base=8,
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    '''
    Modularized VAE encoder.
    '''

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min, fmap_max)

    # Validate arguments.
    assert not is_training or not is_validation

    # Primary inputs.
    reals_in.set_shape([None]+[input_shape])
    reals_in = tf.cast(reals_in, dtype)

    # Encoder network.
    key_ls, size_ls, count_dlatent_size = split_module_names(module_E_list)
    x = reals_in
    for scope_idx, k in enumerate(key_ls):
        if k == 'Standard_E_64':
            x = build_standard_conv_E_64(reals_in=x, name=k, scope_idx=scope_idx)
            break
        elif k == 'Standard_E_128':
            x = build_standard_conv_E_128(reals_in=x, name=k, scope_idx=scope_idx)
            break
        else:
            raise ValueError('Not supported module key:', k)

    # Post-encoder network.
    with tf.variable_scope('ConcatAtts'):
        flat_x = tf.layers.flatten(x)
        e5 = tf.layers.dense(flat_x, 256, activation=tf.nn.relu, name="e5")
        means = tf.layers.dense(e5, latent_size, activation=None, name="means")
        log_var = tf.layers.dense(e5, latent_size, activation=None, name="log_var")

    # Return requested outputs.
    means = tf.identity(means, name='means')
    log_var = tf.identity(log_var, name='log_var')
    return means, log_var
