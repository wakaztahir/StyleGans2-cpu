#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vc2.py
# --- Creation Date: 24-04-2020
# --- Last Modified: Sun 03 May 2020 04:54:55 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss function in VC2.
"""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

def G_logistic_ns(G, D, opt, training_set, minibatch_size, latent_type='uniform'):
    _ = opt
    # latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size, G.input_shapes[0][1]], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size, G.input_shapes[0][1]])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size, G.input_shapes[0][1]])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    return loss, None

def calc_vc_loss(C_delta_latents, regress_out, D_global_size, C_global_size, D_lambda, C_lambda, delta_type):
    assert regress_out.shape.as_list()[1] == (D_global_size + C_global_size)
    # Continuous latents loss
    if delta_type == 'onedim':
        prob_C = tf.nn.softmax(regress_out[:, D_global_size:], axis=1)
        I_loss_C = C_delta_latents * tf.log(prob_C + 1e-12)
        I_loss = C_lambda * I_loss_C

        I_loss_C = tf.reduce_sum(I_loss_C, axis=1)
        I_loss = - I_loss_C
    elif delta_type == 'fulldim':
        I_loss_C = tf.reduce_sum((tf.nn.sigmoid(regress_out[:, D_global_size:]) - C_delta_latents) ** 2, axis=1)
        I_loss = C_lambda * I_loss_C
    return I_loss

def G_logistic_ns_vc2(G, D, I, opt, training_set, minibatch_size, I_info=None, latent_type='uniform',
                     D_global_size=0, D_lambda=0, C_lambda=1, epsilon=0.4,
                     random_eps=False, delta_type='onedim', own_I=False):
    _ = opt
    discrete_latents = None
    C_global_size = G.input_shapes[0][1]-D_global_size
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)
        discrete_latents_2 = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents_2 = tf.one_hot(discrete_latents_2, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    # Sample delta latents
    if delta_type == 'onedim':
        C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
        C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size), latents.dtype)
    elif delta_type == 'fulldim':
        C_delta_latents = tf.random.uniform([minibatch_size, C_global_size], minval=0, maxval=1.0, dtype=latents.dtype)

    if delta_type == 'onedim':
        if not random_eps:
            delta_target = C_delta_latents * epsilon
        else:
            epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=2.0)
            delta_target = C_delta_latents * epsilon
    else:
        delta_target = (C_delta_latents - 0.5) * epsilon

    delta_latents = delta_target + latents

    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)
        delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_latents], axis=1)

    # labels = training_set.get_random_labels_tf(minibatch_size)

    # if own_I:
        # fake1_out, atts = G.get_output_for(latents, labels, is_training=True, return_atts=True)
    # else:
        # fake1_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)
    # fake2_out = G.get_output_for(delta_latents, labels, is_training=True, return_atts=False)

    labels = training_set.get_random_labels_tf(2*minibatch_size)
    latents_all = tf.concat([latents, delta_latents], axis=0)
    fake_all_out, atts_all = G.get_output_for(latents_all, labels, is_training=True, return_atts=True)
    fake1_out, fake2_out = tf.split(fake_all_out, 2, axis=0)
    atts = atts_all[:minibatch_size]

    if I_info is not None:
        fake_scores_out, hidden = D.get_output_for(fake1_out, labels, is_training=True)
    else:
        fake_scores_out = D.get_output_for(fake1_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    
    if own_I:
        regress_out = I.get_output_for(fake1_out, fake2_out, atts, is_training=True)
    else:
        regress_out = I.get_output_for(fake1_out, fake2_out, is_training=True)
    I_loss = calc_vc_loss(C_delta_latents, regress_out, D_global_size, C_global_size, D_lambda, C_lambda, delta_type)
    # I_loss = calc_vc_loss(delta_target, regress_out, D_global_size, C_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    G_loss += I_loss

    return G_loss, None

def D_logistic_r1_vc2(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, latent_type='uniform', D_global_size=0):
    _ = opt, training_set
    discrete_latents = None
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)

    fake_images_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg
