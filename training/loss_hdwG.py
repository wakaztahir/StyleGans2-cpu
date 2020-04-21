#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_hdwG.py
# --- Creation Date: 19-04-2020
# --- Last Modified: Tue 21 Apr 2020 23:57:34 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
HD disentanglement model with trainable G losses.
"""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
from training.loss_hd import calc_vc_loss, calc_cls_loss
from training.loss_hd import reparameterize, log_normal_pdf

def IandMandG_hyperplane_loss(G, D, I, M, opt, training_set, minibatch_size, I_info=None, latent_type='uniform',
               C_global_size=10, D_global_size=0, D_lambda=0, C_lambda=1, cls_alpha=0, epsilon=1,
               random_eps=False, traj_lambda=None, resolution_manual=1024, use_std_in_m=False,
               model_type='hd_dis_model', hyperplane_lambda=1, prior_latent_size=512, hyperdir_lambda=1):
    _ = opt
    resolution_log2 = int(np.log2(resolution_manual))
    nd_out_base = C_global_size // (resolution_log2 - 1)
    nd_out_list = [nd_out_base + C_global_size % (resolution_log2 - 1) if i == 0 else nd_out_base for i in range(resolution_log2 - 1)]
    nd_out_list = nd_out_list[::-1]

    # Sample delta latents
    C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
    C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size), tf.float32)

    delta_var_latents = C_delta_latents

    all_delta_var_latents = tf.eye(C_global_size, dtype=tf.float32)

    labels = training_set.get_random_labels_tf(minibatch_size)

    # Get variation direction in prior latent space.
    prior_var_latents, hyperplane_constraint = M.get_output_for(delta_var_latents, is_training=True)
    prior_all_dirs, _ = M.get_output_for(all_delta_var_latents, is_training=True)

    prior_var_latents = autosummary('Loss/prior_var_latents', prior_var_latents)
    manipulated_prior_dir = tf.matmul(prior_var_latents, tf.transpose(prior_all_dirs)) # [batch, C_global_size]
    manipulated_prior_dir = manipulated_prior_dir * (1. - C_delta_latents) # [batch, C_global_size]
    manipulated_prior_dir = tf.matmul(manipulated_prior_dir, prior_all_dirs) # [batch, prior_latent_size]
    prior_dir_to_go = prior_var_latents - manipulated_prior_dir
    # prior_dir_to_go = prior_var_latents
    prior_dir_to_go = autosummary('Loss/prior_dir_to_go', prior_dir_to_go)

    if latent_type == 'uniform':
        prior_latents = tf.random.uniform([minibatch_size, prior_latent_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        prior_latents = tf.random.normal([minibatch_size, prior_latent_size])
    elif latent_type == 'trunc_normal':
        prior_latents = tf.random.truncated_normal([minibatch_size, prior_latent_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    prior_latents = autosummary('Loss/prior_latents', prior_latents)
    if random_eps:
        epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=2.0)
    prior_delta_latents = prior_latents + epsilon * prior_dir_to_go

    fake1_out = G.get_output_for(prior_latents, labels, is_training=True, randomize_noise=True, normalize_latents=False)
    fake2_out = G.get_output_for(prior_delta_latents, labels, is_training=True, randomize_noise=True, normalize_latents=False)
    fake1_out = autosummary('Loss/fake1_out', fake1_out)

    # Send to D
    fake_scores_out = D.get_output_for(fake1_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    # Send to I
    # regress_out_list = I.get_output_for(fake1_out, fake2_out, is_training=True)
    # regress_out = tf.concat(regress_out_list, axis=1)
    regress_out = I.get_output_for(fake1_out, fake2_out, is_training=True)

    I_loss = calc_vc_loss(C_delta_latents, regress_out, C_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    dir_constraint = tf.reduce_sum(prior_var_latents * prior_dir_to_go, axis=1)
    norm_prior_var_latents = tf.math.sqrt(tf.reduce_sum(prior_var_latents * prior_var_latents, axis=1))
    norm_prior_dir_to_go = tf.math.sqrt(tf.reduce_sum(prior_dir_to_go * prior_dir_to_go, axis=1))
    dir_constraint = - dir_constraint / (norm_prior_var_latents * norm_prior_dir_to_go)
    dir_constraint = autosummary('Loss/dir_constraint', dir_constraint)

    I_loss = I_loss + hyperplane_lambda * hyperplane_constraint + hyperdir_lambda * dir_constraint + G_loss
    # I_loss = I_loss + hyperplane_lambda * hyperplane_constraint + G_loss

    return I_loss, None


def IandG_vc_loss(G, D, I, M, opt, training_set, minibatch_size, I_info=None, latent_type='uniform',
               C_global_size=10, D_global_size=0, D_lambda=0, C_lambda=1, cls_alpha=0, epsilon=1,
               random_eps=False, traj_lambda=None, resolution_manual=1024, use_std_in_m=False,
               model_type='hd_dis_model', hyperplane_lambda=1, prior_latent_size=512, hyperdir_lambda=1):
    _ = opt

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
    C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
    C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size), tf.float32)

    if not random_eps:
        delta_target = C_delta_latents * epsilon
        # delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_target], axis=1)
    else:
        epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=2.0)
        # delta_target = tf.math.abs(C_delta_latents * epsilon)
        delta_target = C_delta_latents * epsilon
        # delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_target], axis=1)
    delta_var_latents = delta_target
    delta_latents = delta_var_latents + latents

    labels = training_set.get_random_labels_tf(minibatch_size)

    # Get variation direction in prior latent space.
    prior_latents = M.get_output_for(latents, is_training=True)
    prior_delta_latents = M.get_output_for(delta_latents, is_training=True)
    prior_delta_latents = autosummary('Loss/prior_delta_latents', prior_delta_latents)

    fake1_out = G.get_output_for(prior_latents, labels, is_training=True, randomize_noise=True, normalize_latents=False)
    fake2_out = G.get_output_for(prior_delta_latents, labels, is_training=True, randomize_noise=True, normalize_latents=False)
    fake1_out = autosummary('Loss/fake1_out', fake1_out)

    # Send to D
    fake_scores_out = D.get_output_for(fake1_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    # Send to I
    regress_out = I.get_output_for(fake1_out, fake2_out, is_training=True)

    I_loss = calc_vc_loss(C_delta_latents, regress_out, C_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    I_loss = I_loss + G_loss

    return I_loss, None
