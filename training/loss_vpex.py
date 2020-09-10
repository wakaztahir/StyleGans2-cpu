#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vpex.py
# --- Creation Date: 07-09-2020
# --- Last Modified: Fri 11 Sep 2020 01:04:30 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss for the extended version of variation predictability model.
"""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary


def calc_vpex_loss(C_delta_latents, regress_out, C_global_size,
                   D_lambda, C_lambda, delta_type):
    assert regress_out.shape.as_list()[1] == C_global_size
    # Continuous latents loss
    # prob_C = tf.nn.softmax(regress_out, axis=1)
    # prob_C = tf.nn.sigmoid(regress_out)
    # I_loss_C = C_delta_latents * tf.log(prob_C + 1e-12)
    # I_loss_C = C_lambda * I_loss_C

    # I_loss_C = tf.reduce_sum(I_loss_C, axis=1)
    # I_loss = -I_loss_C

    # l2_loss
    prob_C = tf.nn.sigmoid(regress_out)
    # prob_C = tf.nn.softmax(regress_out, axis=1)
    I_loss = C_lambda * tf.reduce_sum((prob_C - C_delta_latents)**2, axis=1)

    return I_loss


def G_logistic_ns_vpex(G,
                       D,
                       I,
                       opt,
                       training_set,
                       minibatch_size,
                       latent_type='uniform',
                       D_global_size=0,
                       D_lambda=0,
                       C_lambda=1,
                       epsilon=0.4,
                       random_eps=False,
                       delta_type='onedim',
                       own_I=False):
    _ = opt
    C_global_size = G.input_shapes[0][1]

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] +
                                    [G.input_shapes[0][1]],
                                    minval=-2,
                                    maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] +
                                   [G.input_shapes[0][1]])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal(
            [minibatch_size] + [G.input_shapes[0][1]])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    # Sample delta latents
    C_delta_latents = tf.random.uniform([minibatch_size],
                                        minval=0,
                                        maxval=C_global_size,
                                        dtype=tf.int32)
    C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size),
                              latents.dtype)

    if not random_eps:
        delta_target = C_delta_latents * epsilon
    else:
        epsilon = epsilon * tf.random.normal(
            [minibatch_size, 1], mean=0.0, stddev=2.0)
        delta_target = C_delta_latents * epsilon

    delta_latents = delta_target + latents

    labels = training_set.get_random_labels_tf(2 * minibatch_size)
    latents_all = tf.concat([latents, delta_latents], axis=0)
    fake_all_out = G.get_output_for(latents_all, labels, is_training=True)
    fake1_out, fake2_out = tf.split(fake_all_out, 2, axis=0)

    fake_scores_out = D.get_output_for(fake1_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out)  # -log(sigmoid(fake_scores_out))

    regress_out = I.get_output_for(fake1_out,
                                   fake2_out,
                                   latents,
                                   return_atts=False,
                                   is_training=True)
    I_loss = calc_vpex_loss(C_delta_latents, regress_out,
                            C_global_size, D_lambda, C_lambda, delta_type)
    I_loss = autosummary('Loss/I_loss', I_loss)

    G_loss += I_loss

    return G_loss, None
