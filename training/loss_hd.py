#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_hd.py
# --- Creation Date: 07-04-2020
# --- Last Modified: Sun 12 Apr 2020 16:47:01 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
HD disentanglement model losses.
"""
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

def calc_vc_loss(delta_target, regress_out, C_global_size, D_lambda, C_lambda):
    # Continuous latents loss
    prob_C = tf.nn.softmax(regress_out, axis=1)
    I_loss_C = delta_target * tf.log(prob_C + 1e-12)
    I_loss_C = C_lambda * I_loss_C

    I_loss_C = tf.reduce_sum(I_loss_C, axis=1)
    I_loss = - I_loss_C
    return I_loss

def calc_cls_loss(discrete_latents, cls_out, D_global_size, C_global_size, cls_alpha):
    assert cls_out.shape.as_list()[1] == D_global_size
    prob_D = tf.nn.softmax(cls_out, axis=1)
    I_info_loss_D = discrete_latents * tf.log(prob_D + 1e-12)
    I_info_loss = cls_alpha * I_info_loss_D

    I_info_loss = tf.reduce_sum(I_info_loss, axis=1)
    I_info_loss = - I_info_loss
    return I_info_loss

def IandM_loss(I, M, G, opt, training_set, minibatch_size, I_info=None, latent_type='uniform',
               C_global_size=10, D_global_size=0, D_lambda=0, C_lambda=1, cls_alpha=0, epsilon=3,
               random_eps=False, traj_lambda=None, n_levels=None, resolution_manual=1024):
    _ = opt
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)
        # discrete_latents_2 = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        # discrete_latents_2 = tf.one_hot(discrete_latents_2, D_global_size)

    resolution_log2 = int(np.log2(resolution_manual))
    nd_out_base = C_global_size // (resolution_log2 - 1)
    nd_out_list = [nd_out_base + C_global_size % (resolution_log2 - 1) if i == 0 else nd_out_base for i in range(resolution_log2 - 1)]
    nd_out_list = nd_out_list[::-1]

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size, C_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size, C_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size, C_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    latents = autosummary('Loss/latents', latents)

    # Sample delta latents
    C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=sum(nd_out_list[:n_levels]), dtype=tf.int32)
    C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size), latents.dtype)

    if not random_eps:
        delta_target = C_delta_latents * epsilon
        # delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_target], axis=1)
    else:
        epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=2.0)
        # delta_target = tf.math.abs(C_delta_latents * epsilon)
        delta_target = C_delta_latents * epsilon
        # delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_target], axis=1)

    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)
        # delta_latents = tf.concat([discrete_latents_2, delta_latents], axis=1)
        delta_var_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_target], axis=1)
    else:
        delta_var_latents = delta_target

    delta_latents = delta_var_latents + latents

    labels = training_set.get_random_labels_tf(minibatch_size)

    prior_traj_latents = M.get_output_for(latents, is_training=True)
    prior_traj_latents = autosummary('Loss/prior_traj_latents', prior_traj_latents)
    prior_traj_latents_0 = autosummary('Loss/prior_traj_latents_0', prior_traj_latents[0])
    prior_traj_latents_1 = autosummary('Loss/prior_traj_latents_1', prior_traj_latents[1])
    prior_traj_delta_latents = M.get_output_for(delta_latents, is_training=True)
    fake1_out = G.get_output_for(prior_traj_latents, labels, is_training=True, randomize_noise=True, normalize_latents=False)
    fake2_out = G.get_output_for(prior_traj_delta_latents, labels, is_training=True, randomize_noise=True, normalize_latents=False)
    fake1_out = autosummary('Loss/fake1_out', fake1_out)

    regress_out_list = I.get_output_for(fake1_out, fake2_out, is_training=True)
    regress_out = tf.concat(regress_out_list[:n_levels], axis=1)

    I_loss = calc_vc_loss(C_delta_latents[:,:sum(nd_out_list[:n_levels])], regress_out, C_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    if traj_lambda is not None:
        traj_reg = tf.reduce_sum(prior_traj_latents * prior_traj_latents, axis=1)
        traj_reg = autosummary('Loss/traj_reg', traj_reg)
        I_loss = I_loss + traj_lambda * traj_reg

    if I_info is not None:
        cls_out = I_info.get_output_for(fake1_out, is_training=True)
        I_info_loss = calc_cls_loss(discrete_latents, cls_out, D_global_size, C_global_size, cls_alpha)
        I_info_loss = autosummary('Loss/I_info_loss', I_info_loss)
        I_loss = I_loss + I_info_loss
        I_loss = autosummary('Loss/I_loss_after_INFO', I_loss)

    return I_loss, None
