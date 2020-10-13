#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vc2_subnets_pggan.py
# --- Creation Date: 12-10-2020
# --- Last Modified: Tue 13 Oct 2020 14:44:21 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""

import numpy as np
import tensorflow as tf
from training.utils import get_return_v
from training.vc2_subnets import build_C_spgroup_layers_with_latents_ready

def build_pggan_gen(x, name, n_latents, start_idx, scope_idx, dlatents_in,
                    act, fused_modconv, fmaps=128, resolution=512, fmap_base=2 << 8,
                    fmap_min=1, fmap_max=512, fmap_decay=1,
                    architecture='skip', randomize_noise=True,
                    resample_kernel=[1,3,3,1], num_channels=3,
                    latent_split_ls_for_std_gen=[5,5,5,5],
                    n_subs=4, return_atts=True,
                    pixelnorm_epsilon=1e-8,  # Constant epsilon for pixelwise feature vector normalization.
                    use_pixelnorm=False,  # Enable pixelwise feature vector normalization?
                    use_wscale=True,  # Enable equalized learning rate?
                    **kwargs):
    '''A PGGAN-style network.'''
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    # def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    act = leaky_relu if act == 'lrelu' else tf.nn.relu
    
    dtype = x.dtype
    # latents_in.set_shape([None, latent_size])
    # labels_in.set_shape([None, label_size])
    # combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    combo_in = dlatents_in # [b, latent_size]
    # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    assert n_latents == sum(latent_split_ls_for_std_gen)
    # assert 12 == len(latent_split_ls_for_std_gen)
    latents_ready_spl_ls = []
    for i in range(n_latents):
        with tf.variable_scope('PreConvDense-' + str(i) + '-0'):
            x_tmp0 = dense(dlatents_in[:, i:i+1], fmaps=nf(1))
        with tf.variable_scope('PreConvDense-' + str(i) + '-1'):
            x_tmp1 = dense(x_tmp0, fmaps=nf(1))
        latents_ready_spl_ls.append(x_tmp1[:, tf.newaxis, ...])

    latents_ready_ls = []
    start_code = 0
    for i, seg in enumerate(latent_split_ls_for_std_gen):
        with tf.variable_scope('PreConvConcat-' + str(i)):
            x_tmp = tf.concat(latents_ready_spl_ls[start_code:start_code+seg], axis=1)
        latents_ready_ls.append(x_tmp)
        start_code += seg

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(len(latents_ready_ls) * 2):
        res = (layer_idx + 6) // 2
        shape = [1, 1, 2**res, 2**res]
        noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    def layer(x, layer_idx, up):
        x, atts = get_return_v(build_C_spgroup_layers_with_latents_ready(x, 'SP_latents', latent_split_ls_for_std_gen[layer_idx],
                                                                         layer_idx, latents_ready_ls[layer_idx], return_atts=return_atts,
                                                                         resolution=resolution, n_subs=n_subs, **kwargs), 2)
        if up:
            x = upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)
        else:
            x = conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)

        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)

        x = PN(act(apply_bias(x)))
        return x, atts

    # Building blocks.
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            with tf.variable_scope('Conv0_up'):
                # x, atts_0 = sp_layer(x, layer_idx=res*2-4)
                layer_idx_0 = res*2 - 6
                x, atts_0 = layer(x, layer_idx_0, True)
            with tf.variable_scope('Conv1'):
                layer_idx_1 = res*2 - 5
                x, atts_1 = layer(x, layer_idx_1, False)
        if return_atts:
            atts = tf.concat([atts_0, atts_1], axis=1)
        else:
            atts = None
        return x, atts
    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])

    # x = block(combo_in, 2)
    # images_out = torgb(x, 2)
    atts_out_ls = []
    for res in range(3, resolution_log2 + 1):
        # lod = resolution_log2 - res
        x, atts_tmp = block(x, res)
        atts_out_ls.append(atts_tmp)
        # img = torgb(x, res)
        # images_out = upscale2d(images_out)
        # with tf.variable_scope('Grow_lod%d' % lod):
            # images_out = lerp_clip(img, images_out, lod_in - lod)

    images_out = torgb(x, resolution_log2)

    # assert images_out.dtype == tf.as_dtype(dtype)
    # images_out = tf.identity(images_out, name='images_out')
    # return images_out
    if return_atts:
        with tf.variable_scope('ConcatAtts'):
            atts_out = tf.concat(atts_out_ls, axis=1)
            return tf.identity(images_out, name='images_out'), tf.identity(atts_out, name='atts_out')
    else:
        return tf.identity(images_out, name='images_out')

#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.
