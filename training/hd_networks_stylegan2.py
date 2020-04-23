#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: hd_networks_stylegan2.py
# --- Creation Date: 22-04-2020
# --- Last Modified: Thu 23 Apr 2020 23:31:39 AEST
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
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d

#----------------------------------------------------------------------------
# Main generator network.
# Composed of two sub-networks (mapping and synthesis) that are defined below.
# Used in configs B-F (Table 1).

def G_main(
    latents_in,                                         # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                                          # Second input: Conditioning labels [minibatch, label_size].
    truncation_psi          = 0.5,                      # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = None,                     # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                     # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,                     # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,                    # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,                      # Probability of mixing styles during training. None = disable.
    is_training             = False,                    # Network is under training? Enables and disables specific features.
    is_validation           = False,                    # Network is under validation? Chooses which value to use for truncation_psi.
    return_dlatents         = False,                    # Return dlatents in addition to the images?
    is_template_graph       = False,                    # True = template graph constructed by the Network class, False = actual evaluation.
    components              = dnnlib.EasyDict(),        # Container for sub-networks. Retained between calls.
    mapping_func            = 'G_mapping',              # Build func name for the mapping network.
    synthesis_func          = 'G_synthesis_stylegan2',  # Build func name for the synthesis network.
    **kwargs):                                          # Arguments for sub-networks (mapping and synthesis).

    # Validate arguments.
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=globals()[synthesis_func], **kwargs)
    num_layers = components.synthesis.input_shape[1]
    dlatent_size = components.synthesis.input_shape[2]
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=globals()[mapping_func], dlatent_broadcast=num_layers, **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, is_training=is_training, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if style_mixing_prob is not None:
        with tf.variable_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, is_training=is_training, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

    # Apply truncation trick.
    if truncation_psi is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype=np.float32)
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
            dlatents = tflib.lerp(dlatent_avg, dlatents, layer_psi)

    # Evaluate synthesis network.
    deps = []
    if 'lod' in components.synthesis.vars:
        deps.append(tf.assign(components.synthesis.vars['lod'], lod_in))
    with tf.control_dependencies(deps):
        images_out = components.synthesis.get_output_for(dlatents, is_training=is_training, force_clean_graph=is_template_graph, **kwargs)

    # Return requested outputs.
    images_out = tf.identity(images_out, name='images_out')
    if return_dlatents:
        return images_out, dlatents
    return images_out

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
            # with tf.variable_scope('Bottleneck'):
                # tmp = networks_stylegan2.apply_bias_act(networks_stylegan2.dense_layer(x[:, start_tmp:end_tmp], fmaps=mapping_fmaps, lrmul=mapping_lrmul),
                                     # act=act, lrmul=mapping_lrmul)
            # with tf.variable_scope('to_dlatent'):
                # x_list.append(networks_stylegan2.apply_bias_act(networks_stylegan2.dense_layer(tmp, fmaps=dlatent_size, lrmul=mapping_lrmul),
                                             # act=act, lrmul=mapping_lrmul))
            with tf.variable_scope('to_dlatent'):
                x_list.append(x[:, start_tmp:end_tmp])
    x = tf.stack(x_list, axis=1)

    # # Broadcast.
    # if dlatent_broadcast is not None:
        # with tf.variable_scope('Broadcast'):
            # x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

#----------------------------------------------------------------------------
# StyleGAN synthesis network with revised architecture (Figure 2d).
# Implements progressive growing, but no skip connections or residual nets (Figure 7).
# Used in configs B-D (Table 1).

def G_synthesis_stylegan_revised_hd(
    dlatents_in,                        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    structure           = 'auto',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    force_clean_graph   = False,        # True = construct a clean graph that looks nice in TensorBoard, False = default behavior.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    if is_template_graph: force_clean_graph = True
    if force_clean_graph: randomize_noise = False
    if structure == 'auto': structure = 'linear' if force_clean_graph else 'recursive'
    act = nonlinearity
    num_layers = resolution_log2 * 2 - 2
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_layers, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2**res, 2**res]
        noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, fmaps, kernel, up=False):
        x = networks_stylegan2.modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up, resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        if layer_idx > num_layers * 0.75:
            if randomize_noise:
                noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
            else:
                noise = tf.cast(noise_inputs[layer_idx], x.dtype)
            noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
            x += noise * tf.cast(noise_strength, x.dtype)
        return networks_stylegan2.apply_bias_act(x, act=act)

    # Early layers.
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)

    # Building blocks for remaining layers.
    def block(res, x): # res = 3..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            with tf.variable_scope('Conv0_up'):
                x = layer(x, layer_idx=res*2-5, fmaps=nf(res-1), kernel=3, up=True)
            with tf.variable_scope('Conv1'):
                x = layer(x, layer_idx=res*2-4, fmaps=nf(res-1), kernel=3)
            return x
    def torgb(res, x): # res = 2..resolution_log2
        with tf.variable_scope('ToRGB_lod%d' % (resolution_log2 - res)):
            return networks_stylegan2.apply_bias_act(networks_stylegan2.modulated_conv2d_layer(x, dlatents_in[:, res*2-3], fmaps=num_channels, kernel=1, demodulate=False, fused_modconv=fused_modconv))

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        for res in range(3, resolution_log2 + 1):
            x = block(res, x)
        images_out = torgb(resolution_log2, x)

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        images_out = torgb(2, x)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(res, x)
            img = torgb(res, x)
            with tf.variable_scope('Upsample_lod%d' % lod):
                images_out = upsample_2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = tflib.lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
        def grow(x, res, lod):
            y = block(res, x)
            img = lambda: networks_stylegan2.naive_upsample_2d(torgb(res, y), factor=2**lod)
            img = cset(img, (lod_in > lod), lambda: networks_stylegan2.naive_upsample_2d(tflib.lerp(torgb(res, y), upsample_2d(torgb(res - 1, x)), lod_in - lod), factor=2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(x, 3, resolution_log2 - 3)

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')
