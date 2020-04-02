#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vid.py
# --- Creation Date: 24-03-2020
# --- Last Modified: Thu 02 Apr 2020 00:28:34 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Vid Losses
"""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

def calc_cls_loss(discrete_latents, cls_out, D_global_size, C_global_size, cls_alpha):
    assert cls_out.shape.as_list()[1] == D_global_size
    prob_D = tf.nn.softmax(cls_out, axis=1)
    I_info_loss_D = tf.reduce_sum(discrete_latents * tf.log(prob_D + 1e-12), axis=1)
    I_info_loss = - cls_alpha * I_info_loss_D
    return I_info_loss

def calc_mi(t_out, t_out_margin, log_scope):
    t1 = tf.reduce_mean(t_out)
    # t1 = autosummary('Loss/'+log_scope+'/t1', t1)
    t2 = tf.math.log(tf.reduce_mean(tf.math.exp(t_out_margin)) + 1e-12)
    # t2 = autosummary('Loss/'+log_scope+'/t2', t2)
    v = tf.math.subtract(t1, t2)
    return v

def calc_vid_pair_loss(mi_l, mi_h, C_lambda):
    loss = tf.math.square(mi_h - mi_l)
    return C_lambda * loss

def calc_vc_loss(C_delta_latents, regress_out, D_global_size, C_global_size,
                 D_lambda, C_lambda):
    assert regress_out.shape.as_list()[1] == (D_global_size + C_global_size)
    # Continuous latents loss
    prob_C = tf.nn.softmax(regress_out[:, D_global_size:], axis=1)
    I_loss_C = C_delta_latents * tf.log(prob_C + 1e-12)
    I_loss = C_lambda * I_loss_C

    I_loss_C = tf.reduce_sum(I_loss_C, axis=1)
    I_loss = - I_loss_C
    return I_loss

def G_logistic_ns_vid(G, D, I, opt, training_set, minibatch_size, I_info=None, latent_type='uniform',
                     D_global_size=0, D_lambda=0, C_lambda=1, cls_alpha=0):
    _ = opt
    discrete_latents = None
    C_global_size = G.input_shapes[0][1]-D_global_size
    C_global_size = autosummary('Loss/G_c_size', C_global_size)
    minibatch_size = autosummary('Loss/G_minibatch_size', minibatch_size)
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)
        discrete_latents_2 = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents_2 = tf.one_hot(discrete_latents_2, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2., maxval=2.)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    latents = autosummary('Loss/G_latents', latents)

    # Sample delta latents
    C_delta_idxes = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
    C_delta_latents = tf.cast(tf.one_hot(C_delta_idxes, C_global_size), latents.dtype)
    C_delta_idxes_2 = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
    C_delta_latents_2 = tf.cast(tf.one_hot(C_delta_idxes_2, C_global_size), latents.dtype)
    C_delta_idxes_3 = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
    C_delta_latents_3 = tf.cast(tf.one_hot(C_delta_idxes_3, C_global_size), latents.dtype)
    C_delta_latents = autosummary('Loss/G_delta_latents_1', C_delta_latents)
    C_delta_latents_2 = autosummary('Loss/G_delta_latents_2', C_delta_latents_2)
    C_delta_latents_3 = autosummary('Loss/G_delta_latents_3', C_delta_latents_3)

    # Sample n_frames_low and _high
    n_frames_l = tf.random.uniform([], minval=2, maxval=5, dtype=tf.int32)
    # n_frames_l = tf.constant(3, dtype=tf.int32)
    n_frames_h = tf.random.uniform([], minval=5, maxval=8, dtype=tf.int32)
    # n_frames_h = tf.constant(3, dtype=tf.int32)

    # Produce masks for low and high
    mask_shared = tf.expand_dims(C_delta_latents, 1)
    tile_shape_l = tf.convert_to_tensor([1, n_frames_l, 1], dtype=tf.int32)
    tile_shape_h = tf.convert_to_tensor([1, n_frames_h, 1], dtype=tf.int32)
    mask_l = tf.tile(mask_shared, tile_shape_l) > 0
    mask_h = tf.tile(mask_shared, tile_shape_h) > 0

    # Duplicate base latents
    latents_expdim = tf.expand_dims(latents, 1)
    latents_l = tf.tile(latents_expdim, tile_shape_l)
    latents_h = tf.tile(latents_expdim, tile_shape_h)

    # Generate linspace for the delta dimension with randomness
    dim_i_l = tf.linspace(-1., 1., n_frames_l, name='linspace_latent_low')
    dim_i_l_delta = tf.random.uniform([n_frames_l], 
                                      minval=(-2.)/tf.cast(n_frames_l, tf.float32), 
                                      maxval=2./tf.cast(n_frames_l, tf.float32))
    dim_i_l = dim_i_l + dim_i_l_delta
    dim_i_l = tf.reshape(dim_i_l, [1, n_frames_l, 1])
    dim_i_l = tf.tile(dim_i_l, [minibatch_size, 1, C_global_size])
    dim_i_h = tf.linspace(-2., 2., n_frames_h, name='linspace_latent_high')
    dim_i_h_delta = tf.random.uniform([n_frames_h], 
                                      minval=(-2.)/tf.cast(n_frames_h, tf.float32), 
                                      maxval=2./tf.cast(n_frames_h, tf.float32))
    dim_i_h = dim_i_h + dim_i_h_delta
    dim_i_h = tf.reshape(dim_i_h, [1, n_frames_h, 1])
    dim_i_h = tf.tile(dim_i_h, [minibatch_size, 1, C_global_size])

    # Assign linspace with masks
    latents_l = tf.where(mask_l, dim_i_l, latents_l) # [b, n_frames_l, n_cdim]
    latents_h = tf.where(mask_h, dim_i_h, latents_h) # [b, n_frames_h, n_cdim]

    if D_global_size > 0:
        # Duplicate discrete latents
        d_latents_expdim = tf.expand_dims(discrete_latents, 1)
        d_latents_l = tf.tile(d_latents_expdim, tile_shape_l) # [b, n_frames_l, n_ddim]
        d_latents_h = tf.tile(d_latents_expdim, tile_shape_h) # [b, n_frames_h, n_ddim]

    # First deal with plain latents with loss_from_D
    # Concat discrete latents
    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_out = G.get_output_for(latents, labels, is_training=True)
    if I_info is not None:
        fake_scores_out, hidden = D.get_output_for(fake_out, labels, is_training=True)
        cls_out = I_info.get_output_for(hidden, is_training=True)
        I_info_loss = calc_cls_loss(discrete_latents, cls_out, D_global_size,
                                    G.input_shapes[0][1]-D_global_size, cls_alpha)
        I_info_loss = autosummary('Loss/I_info_loss', I_info_loss)
    else:
        fake_scores_out = D.get_output_for(fake_out, labels, is_training=True)
        I_info_loss = None
    fake_scores_out = autosummary('Loss/G_fake_scores_out', fake_scores_out)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    # Second deal with vid latents with loss_from_I
    # Concat discrete latents
    if D_global_size > 0:
        latents_l = tf.concat([d_latents_l, latents_l], axis=2) # [b, n_frames_l, n_ddim+n_cdim]
        latents_l = tf.reshape(latents_l, [minibatch_size * n_frames_l, D_global_size + C_global_size])
        latents_h = tf.concat([d_latents_h, latents_h], axis=2) # [b, n_frames_h, n_ddim+n_cdim]
        latents_h = tf.reshape(latents_h, [minibatch_size * n_frames_h, D_global_size + C_global_size])
    else:
        latents_l = tf.reshape(latents_l, [minibatch_size * n_frames_l, C_global_size])
        latents_h = tf.reshape(latents_h, [minibatch_size * n_frames_h, C_global_size])
    labels_l = training_set.get_random_labels_tf(minibatch_size * n_frames_l)
    labels_h = training_set.get_random_labels_tf(minibatch_size * n_frames_h)
    fake_out_l = G.get_output_for(latents_l, labels_l, is_training=True)
    fake_out_h = G.get_output_for(latents_h, labels_h, is_training=True)
    fake_out_l_shape = tf.shape(fake_out_l) # [b*n_frames_l, c, h, w]
    fake_out_h_shape = tf.shape(fake_out_h) # [b*n_frames_h, c, h, w]
    fake_out_l = tf.reshape(fake_out_l, [minibatch_size, -1, fake_out_l_shape[1],
                                         fake_out_l_shape[2], fake_out_l_shape[3]]) # [b, n_frames_l, c, h, w]
    fake_out_l = tf.transpose(fake_out_l, [0, 2, 1, 3, 4]) # [b, c, n_frames_l, h, w]
    fake_out_h = tf.reshape(fake_out_h, [minibatch_size, -1, fake_out_h_shape[1],
                                         fake_out_h_shape[2], fake_out_h_shape[3]]) # [b, n_frames_h, c, h, w]
    fake_out_h = tf.transpose(fake_out_h, [0, 2, 1, 3, 4]) # [b, c, n_frames_h, h, w]
    fake_out_l = autosummary('Loss/G_fake_out_l', fake_out_l)
    fake_out_h = autosummary('Loss/G_fake_out_h', fake_out_h)

    t_out_l = I.get_output_for(fake_out_l, C_delta_latents, is_training=True)
    t_out_l_margin = I.get_output_for(fake_out_l, C_delta_latents_2, is_training=True)

    t_out_h = I.get_output_for(fake_out_h, C_delta_latents, is_training=True)
    t_out_h_margin = I.get_output_for(fake_out_h, C_delta_latents_3, is_training=True)
    t_out_h = autosummary('Loss/G_t_out_h', t_out_h)
    t_out_h_margin = autosummary('Loss/G_t_out_h_margin', t_out_h_margin)

    # Calculate Vid loss
    t1_l = -tf.nn.softplus(-t_out_l)
    # t2_l = tf.math.log(tf.reduce_mean(tf.math.exp(t_out_l_margin)))
    t2_l = tf.nn.softplus(t_out_l_margin)
    mi_l = t1_l - t2_l
    t1_h = -tf.nn.softplus(-t_out_h)
    t1_h = autosummary('Loss/G_t1_h', t1_h)
    # t2_h = tf.math.log(tf.reduce_mean(tf.math.exp(t_out_h_margin)))
    t2_h = tf.nn.softplus(t_out_h_margin)
    t2_h = autosummary('Loss/G_t2_h', t2_h)
    mi_h = t1_h - t2_h
    mi_h = autosummary('Loss/G_mi_h', mi_h)
    g_vid_loss_raw = tf.math.square(mi_h - mi_l)
    g_vid_loss = C_lambda * g_vid_loss_raw

    return G_loss, None, g_vid_loss, I_info_loss

def G_logistic_ns_vid_naive_cluster(G, D, I, opt, training_set, minibatch_size, 
                                    I_info=None, latent_type='uniform', D_global_size=0, 
                                    D_lambda=0, C_lambda=1, cls_alpha=0):
    _ = opt
    discrete_latents = None
    C_global_size = G.input_shapes[0][1]-D_global_size
    C_global_size = autosummary('Loss/G_c_size', C_global_size)
    minibatch_size = autosummary('Loss/G_minibatch_size', minibatch_size)
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2., maxval=2.)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    latents = autosummary('Loss/G_latents', latents)

    # Sample delta latents
    C_delta_idxes = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
    C_delta_latents = tf.cast(tf.one_hot(C_delta_idxes, C_global_size), latents.dtype)

    # Sample n_frames_high
    # n_frames_h = tf.random.uniform([], minval=2, maxval=8, dtype=tf.int32)
    n_frames_h = tf.constant(3, dtype=tf.int32)

    # Produce masks for high
    mask_shared = tf.expand_dims(C_delta_latents, 1)
    tile_shape_h = tf.convert_to_tensor([1, n_frames_h, 1], dtype=tf.int32)
    mask_h = tf.tile(mask_shared, tile_shape_h) > 0

    # Duplicate base latents
    latents_expdim = tf.expand_dims(latents, 1)
    latents_h = tf.tile(latents_expdim, tile_shape_h)

    # Generate linspace for the delta dimension with randomness
    dim_i_h = tf.linspace(-2., 2., n_frames_h, name='linspace_latent_high')
    dim_i_h_delta = tf.random.uniform([n_frames_h], 
                                      minval=(-2.)/tf.cast(n_frames_h, tf.float32), 
                                      maxval=2./tf.cast(n_frames_h, tf.float32))
    dim_i_h = dim_i_h + dim_i_h_delta
    dim_i_h = tf.reshape(dim_i_h, [1, n_frames_h, 1])
    dim_i_h = tf.tile(dim_i_h, [minibatch_size, 1, C_global_size])

    # Assign linspace with masks
    latents_h = tf.where(mask_h, dim_i_h, latents_h) # [b, n_frames_h, n_cdim]

    if D_global_size > 0:
        # Duplicate discrete latents
        d_latents_expdim = tf.expand_dims(discrete_latents, 1)
        d_latents_h = tf.tile(d_latents_expdim, tile_shape_h) # [b, n_frames_h, n_ddim]

    # First deal with plain latents with loss_from_D
    # Concat discrete latents
    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_out = G.get_output_for(latents, labels, is_training=True)
    if I_info is not None:
        fake_scores_out, hidden = D.get_output_for(fake_out, labels, is_training=True)
        cls_out = I_info.get_output_for(hidden, is_training=True)
        I_info_loss = calc_cls_loss(discrete_latents, cls_out, D_global_size,
                                    G.input_shapes[0][1]-D_global_size, cls_alpha)
        I_info_loss = autosummary('Loss/I_info_loss', I_info_loss)
    else:
        fake_scores_out = D.get_output_for(fake_out, labels, is_training=True)
        I_info_loss = None
    fake_scores_out = autosummary('Loss/G_fake_scores_out', fake_scores_out)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    # Second deal with vid latents with loss_from_I
    # Concat discrete latents
    if D_global_size > 0:
        latents_h = tf.concat([d_latents_h, latents_h], axis=2) # [b, n_frames_h, n_ddim+n_cdim]
        latents_h = tf.reshape(latents_h, 
                               [minibatch_size * n_frames_h, D_global_size + C_global_size])
    else:
        latents_h = tf.reshape(latents_h, [minibatch_size * n_frames_h, C_global_size])
    labels_h = training_set.get_random_labels_tf(minibatch_size * n_frames_h)
    fake_out_h = G.get_output_for(latents_h, labels_h, is_training=True)
    fake_out_h_shape = tf.shape(fake_out_h) # [b*n_frames_h, c, h, w]
    fake_out_h = tf.reshape(fake_out_h, [minibatch_size, -1, fake_out_h_shape[1],
                                         fake_out_h_shape[2], fake_out_h_shape[3]]) # [b, n_frames_h, c, h, w]
    fake_out_h = tf.transpose(fake_out_h, [0, 2, 1, 3, 4]) # [b, c, n_frames_h, h, w]
    fake_out_h = autosummary('Loss/G_fake_out_h', fake_out_h)

    regress_out = I.get_output_for(fake_out_h, is_training=True)
    regress_out = autosummary('Loss/G_regress_out', regress_out)

    I_loss = calc_vc_loss(C_delta_latents, regress_out, D_global_size,
                          C_global_size, D_lambda, C_lambda)
    # I_loss = calc_vc_loss(delta_target, regress_out, D_global_size, C_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    return G_loss, None, I_loss, I_info_loss

def I_vid(G, I, opt, training_set, minibatch_size, latent_type='uniform',
          D_global_size=0, D_lambda=0, C_lambda=1, MI_lambda=1):
    _ = opt
    discrete_latents = None
    C_global_size = G.input_shapes[0][1]-D_global_size
    C_global_size = autosummary('Loss/I_c_size', C_global_size)
    minibatch_size = autosummary('Loss/I_minibatch_size', minibatch_size)
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2., maxval=2.)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    latents = autosummary('Loss/I_latents', latents)

    # Sample delta latents
    C_delta_idxes = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
    C_delta_latents = tf.cast(tf.one_hot(C_delta_idxes, C_global_size), latents.dtype)
    C_delta_idxes_2 = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
    C_delta_latents_2 = tf.cast(tf.one_hot(C_delta_idxes_2, C_global_size), latents.dtype)
    C_delta_latents = autosummary('Loss/I_delta_latents_1', C_delta_latents)
    C_delta_latents_2 = autosummary('Loss/I_delta_latents_2', C_delta_latents_2)

    # Sample n_frames
    n_frames = tf.random.uniform([], minval=2, maxval=8, dtype=tf.int32)
    # n_frames = tf.constant(3, dtype=tf.int32)

    # Produce mask
    mask_shared = tf.expand_dims(C_delta_latents, 1)
    tile_shape = tf.convert_to_tensor([1, n_frames, 1], dtype=tf.int32)
    mask = tf.tile(mask_shared, tile_shape) > 0

    # Duplicate base latents
    latents_expdim = tf.expand_dims(latents, 1)
    latents = tf.tile(latents_expdim, tile_shape)

    # Generate linspace for the delta dimension with randomness
    dim_i = tf.linspace(-2., 2., n_frames, name='linspace_latent')
    dim_i_delta = tf.random.uniform([n_frames],
                                    minval=(-2.)/tf.cast(n_frames, tf.float32),
                                    maxval=2./tf.cast(n_frames, tf.float32))
    dim_i = dim_i + dim_i_delta
    dim_i = tf.reshape(dim_i, [1, n_frames, 1])
    dim_i = tf.tile(dim_i, [minibatch_size, 1, C_global_size])

    # Assign linspace with masks
    latents = tf.where(mask, dim_i, latents) # [b, n_frames, n_cdim]

    if D_global_size > 0:
        # Duplicate discrete latents
        d_latents_expdim = tf.expand_dims(discrete_latents, 1)
        d_latents = tf.tile(d_latents_expdim, tile_shape) # [b, n_frames, n_ddim]
        # Concat discrete latents
        latents = tf.concat([d_latents, latents], axis=2) # [b, n_frames, n_ddim+n_cdim]
        latents = tf.reshape(latents, [minibatch_size * n_frames, D_global_size + C_global_size])
    else:
        latents = tf.reshape(latents, [minibatch_size * n_frames, C_global_size])

    labels = training_set.get_random_labels_tf(minibatch_size * n_frames)
    fake_out = G.get_output_for(latents, labels, is_training=True)
    fake_out_shape = tf.shape(fake_out) # [b*n_frames, c, h, w]
    fake_out = tf.reshape(fake_out, [minibatch_size, -1, fake_out_shape[1],
                                     fake_out_shape[2], fake_out_shape[3]])
    fake_out = tf.transpose(fake_out, [0, 2, 1, 3, 4])
    fake_out = autosummary('Loss/I_fake_out', fake_out)
    t_out = I.get_output_for(fake_out, C_delta_latents, is_training=True)
    t_out = autosummary('Loss/I_t_out', t_out)
    t_out_margin = I.get_output_for(fake_out, C_delta_latents_2, is_training=True)
    t_out_margin = autosummary('Loss/I_t_out_margin', t_out_margin)

    # Calculate MINE loss
    t1 = -tf.nn.softplus(-t_out)
    t1 = autosummary('Loss/I_t1', t1)
    # t2 = tf.math.log(tf.reduce_mean(tf.math.exp(t_out_margin)))
    t2 = tf.nn.softplus(t_out_margin)
    t2 = autosummary('Loss/I_t2', t2)
    mi_out = t1 - t2
    mi_out = autosummary('Loss/I_raw_mi_out', mi_out)
    I_loss = -(mi_out * MI_lambda)
    return I_loss, None

def I_vid_blurry(G, I, opt, training_set, minibatch_size, blurry_fake_out_mem, latent_type='uniform',
                 D_global_size=0, D_lambda=0, C_lambda=1, MI_lambda=1, phi=0.5):
    _ = opt
    discrete_latents = None
    C_global_size = G.input_shapes[0][1]-D_global_size
    C_global_size = autosummary('Loss/I_c_size', C_global_size)
    minibatch_size = autosummary('Loss/I_minibatch_size', minibatch_size)
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2., maxval=2.)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    latents = autosummary('Loss/I_latents', latents)

    # Sample delta latents
    C_delta_idxes = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
    C_delta_latents = tf.cast(tf.one_hot(C_delta_idxes, C_global_size), latents.dtype)

    # Sample n_frames
    # n_frames = tf.random.uniform([], minval=2, maxval=8, dtype=tf.int32)
    n_frames = tf.constant(3, dtype=tf.int32)

    # Produce mask
    mask_shared = tf.expand_dims(C_delta_latents, 1)
    tile_shape = tf.convert_to_tensor([1, n_frames, 1], dtype=tf.int32)
    mask = tf.tile(mask_shared, tile_shape) > 0

    # Duplicate base latents
    latents_expdim = tf.expand_dims(latents, 1)
    latents = tf.tile(latents_expdim, tile_shape)

    # Generate linspace for the delta dimension with randomness
    dim_i = tf.linspace(-2., 2., n_frames, name='linspace_latent')
    dim_i_delta = tf.random.uniform([n_frames],
                                    minval=(-2.)/tf.cast(n_frames, tf.float32),
                                    maxval=2./tf.cast(n_frames, tf.float32))
    dim_i = dim_i + dim_i_delta
    dim_i = tf.reshape(dim_i, [1, n_frames, 1])
    dim_i = tf.tile(dim_i, [minibatch_size, 1, C_global_size])

    # Assign linspace with masks
    latents = tf.where(mask, dim_i, latents) # [b, n_frames, n_cdim]

    if D_global_size > 0:
        # Duplicate discrete latents
        d_latents_expdim = tf.expand_dims(discrete_latents, 1)
        d_latents = tf.tile(d_latents_expdim, tile_shape) # [b, n_frames, n_ddim]
        # Concat discrete latents
        latents = tf.concat([d_latents, latents], axis=2) # [b, n_frames, n_ddim+n_cdim]
        latents = tf.reshape(latents, [minibatch_size * n_frames, D_global_size + C_global_size])
    else:
        latents = tf.reshape(latents, [minibatch_size * n_frames, C_global_size])

    labels = training_set.get_random_labels_tf(minibatch_size * n_frames)
    fake_out = G.get_output_for(latents, labels, is_training=True)
    fake_out_shape = tf.shape(fake_out) # [b*n_frames, c, h, w]
    fake_out = tf.reshape(fake_out, [minibatch_size, -1, fake_out_shape[1],
                                     fake_out_shape[2], fake_out_shape[3]])
    fake_out = tf.transpose(fake_out, [0, 2, 1, 3, 4])
    fake_out = autosummary('Loss/I_fake_out', fake_out)
    blurry_fake_out = (1 - phi) * fake_out + phi * blurry_fake_out_mem
    regress_out = I.get_output_for(blurry_fake_out, is_training=True)
    regress_out = autosummary('Loss/I_regress_out', regress_out)

    I_loss = calc_vc_loss(C_delta_latents, regress_out, D_global_size,
                          C_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    return I_loss, blurry_fake_out, None

def D_logistic_r1_vid(G, D, opt, training_set, minibatch_size, reals, labels,
                      gamma=10.0, latent_type='uniform', D_global_size=0):
    _ = opt, training_set
    discrete_latents = None
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2., maxval=2.)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)

    fake_images_out = G.get_output_for(latents, labels, is_training=True)
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

def D_logistic_r1_info_gan_vid(G, D, opt, training_set, minibatch_size, reals, 
                               labels, gamma=10.0, latent_type='uniform', D_global_size=0):
    _ = opt, training_set
    discrete_latents = None
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2., maxval=2.)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)

    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, _ = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out, _ = D.get_output_for(fake_images_out, labels, is_training=True)
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
