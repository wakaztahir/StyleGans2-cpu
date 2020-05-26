#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_inference.py
# --- Creation Date: 27-05-2020
# --- Last Modified: Wed 27 May 2020 01:35:33 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss function for inference network training.
"""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary


def I_loss(G, I, opt, minibatch_size, latent_type='normal', dlatent_size=10):
    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size, dlatent_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size, dlatent_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size, dlatent_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    fakes = G.get_output_for(latents, None, is_training=True, return_atts=False)
    regress_out = I.get_output_for(fakes, is_training=True)
    loss = tf.reduce_sum(tf.math.squared_difference(latents, regress_out), axis=1)
    return loss, None
