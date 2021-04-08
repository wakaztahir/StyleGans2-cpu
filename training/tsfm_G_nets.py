#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: tsfm_G_nets.py
# --- Creation Date: 05-04-2021
# --- Last Modified: Thu 08 Apr 2021 16:10:32 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Transformer Generator Networks.
"""

import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
from training.modular_networks2 import torgb
from training.modular_networks2 import split_module_names
from training.modular_networks2 import build_Const_layers
from training.modular_networks2 import build_C_global_layers
from training.modular_networks2 import build_noise_layer, build_conv_layer
from training.modular_networks2 import build_res_conv_layer
from training.modular_networks2 import build_C_spgroup_layers
from training.modular_networks2 import build_C_spgroup_softmax_layers
# from training.modular_transformer import build_trans_cond_layer
from training.modular_transformer import build_trans_z_to_mask_layer
from training.modular_transformer import build_trans_pos_to_mask_layer
from training.modular_transformer import build_trans_mask_to_feat_layer
from training.modular_transformer import build_trans_mask_to_feat_encoder_layer
from training.utils import get_return_v

#----------------------------------------------------------------------------
# PS-SC main Generator
def G_main_tsfm(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        components=dnnlib.EasyDict(
        ),  # Container for sub-networks. Retained between calls.
        mapping_func='G_mapping_tsfm',  # Build func name for the mapping network.
        synthesis_func='G_synthesis_modular_tsfm',  # Build func name for the synthesis network.
        return_atts=False,  # If return atts.
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    # Validate arguments.
    assert not is_training or not is_validation

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network(
            'G_synthesis', func_name=globals()[synthesis_func], return_atts=return_atts, **kwargs)
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=globals()[mapping_func],
                                           dlatent_broadcast=None, **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, is_training=is_training, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Evaluate synthesis network.
    deps = []
    if 'lod' in components.synthesis.vars:
        deps.append(tf.assign(components.synthesis.vars['lod'], lod_in))
    with tf.control_dependencies(deps):
        if return_atts:
            images_out, atts_out = components.synthesis.get_output_for(dlatents, is_training=is_training,
                                                             force_clean_graph=is_template_graph, return_atts=True, **kwargs)
        else:
            images_out = components.synthesis.get_output_for(dlatents, is_training=is_training,
                                                             force_clean_graph=is_template_graph, return_atts=False, **kwargs)

    # Return requested outputs.
    images_out = tf.identity(images_out, name='images_out')
    if return_atts:
        atts_out = tf.identity(atts_out, name='atts_out')
        return images_out, atts_out
    else:
        return images_out


def G_mapping_tsfm(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        latent_size=7,  # Latent vector (Z) dimensionality.
        label_size=0,  # Label dimensionality, 0 if no labels.
        mapping_nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        **_kwargs):  # Ignore unrecognized keyword args.

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    if label_size > 0:
        with tf.variable_scope('LabelConcat'):
            x = tf.concat([labels_in, x], axis=1)

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

def G_synthesis_modular_tsfm(
        dlatents_in,  # Input: Disentangled latents (W) [minibatch, label_size+dlatent_size].
        dlatent_size=7,  # Disentangled latent (W) dimensionality. Including discrete info, rotation, scaling, xy shearing, and xy translation.
        label_size=0,  # Label dimensionality, 0 if no labels.
        module_list=None,  # A list containing module names, which represent semantic latents (exclude labels).
        num_channels=1,  # Number of output color channels.
        resolution=128,  # Output resolution.
        architecture='skip', # Architecture: 'orig', 'skip', 'resnet'.
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        use_noise=False,  # If noise is used in this dataset.
        randomize_noise=True,  # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        return_atts=False,  # If return atts.
        G_nf_scale=4,
        trans_dim=512,  # Transformer dimension.
        is_training=True,  # If the model is in training mode.
        post_trans_wh=16,  # The cnn h and w after transformer.
        post_trans_cnn_dim=128,  # The cnn fmap after transformer.
        dff=512,  # The dff in transformers.
        trans_rate=0.1,  # The dropout rate in transformers.
        **kwargs):  # Ignore unrecognized keyword args.
    '''
    Modularized Transformer network.
    '''

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min, fmap_max)

    act = nonlinearity
    images_out = None

    # Note that module_list may include modules not containing latents,
    # e.g. Conv layers (size in this case means number of conv layers).
    key_ls, size_ls, count_dlatent_size = split_module_names(module_list)

    # Primary inputs.
    assert dlatent_size == count_dlatent_size
    dlatents_in.set_shape([None, count_dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Early layers consists of 4x4 constant layer.
    y = None

    subkwargs = EasyDict()
    subkwargs.update(dlatents_in=dlatents_in, act=act, dtype=dtype, resample_kernel=resample_kernel,
                     fused_modconv=fused_modconv, use_noise=use_noise, randomize_noise=randomize_noise,
                     resolution=resolution, fmap_base=fmap_base, architecture=architecture,
                     num_channels=num_channels, fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, 
                     dff=dff, trans_rate=trans_rate, is_training=is_training, **kwargs)

    # Build modules by module_dict.
    start_idx = 0
    x = dlatents_in
    atts = []
    noise_inputs = []
    for scope_idx, k in enumerate(key_ls):
        if k.startswith('Trans_z2mask-'):
            # e.g. {'Trans_z2mask-3-1': 10} (format: name-n_layers-n_subs)
            n_subs = int(k.split('-')[-1])
            n_layers = int(k.split('-')[-2])
            if return_atts:
                x, atts_tmp = build_trans_z_to_mask_layer(x, name=k, n_layers=n_layers,
                                                          scope_idx=scope_idx, wh=post_trans_wh, n_subs=n_subs,
                                                          trans_dim=trans_dim, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = get_return_v(build_trans_z_to_mask_layer(x, name=k, n_layers=n_layers,
                                                             scope_idx=scope_idx, wh=post_trans_wh, n_subs=n_subs,
                                                             trans_dim=trans_dim, **subkwargs), 1)
        elif k.startswith('Trans_pos2mask-'):
            # e.g. {'Trans_pos2mask-3-1': 10} (format: {name-n_layers-n_subs: n_masks (nlatents)})
            n_subs = int(k.split('-')[-1])
            n_layers = int(k.split('-')[-2])
            if return_atts:
                x, atts_tmp = build_trans_pos_to_mask_layer(x, name=k, n_layers=n_layers,
                                                            scope_idx=scope_idx, wh=post_trans_wh, n_subs=n_subs,
                                                            trans_dim=trans_dim, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = get_return_v(build_trans_pos_to_mask_layer(x, name=k, n_layers=n_layers,
                                                               scope_idx=scope_idx, wh=post_trans_wh, n_subs=n_subs,
                                                               trans_dim=trans_dim, **subkwargs), 1)
        elif k == 'Trans_mask2feat':
            # e.g. {'Trans_mask2feat': 2}
            x = build_trans_mask_to_feat_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                               wh=post_trans_wh, feat_cnn_dim=post_trans_cnn_dim,
                                               trans_dim=trans_dim, **subkwargs)
        elif k == 'Trans_mask2feat_enc':
            # e.g. {'Trans_mask2feat': 2}
            x = build_trans_mask_to_feat_encoder_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                                       wh=post_trans_wh, feat_cnn_dim=post_trans_cnn_dim,
                                                       trans_dim=trans_dim, **subkwargs)
        elif k == 'Noise':
            # e.g. {'Noise': 1}
            # print('out noise_inputs:', noise_inputs)
            x = build_noise_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                  fmaps=nf(scope_idx//G_nf_scale), noise_inputs=noise_inputs, **subkwargs)
        elif k == 'ResConv-id' or k == 'ResConv-up' or k == 'ResConv-down':
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_res_conv_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                     fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
        elif k == 'Conv-id' or k == 'Conv-up' or k == 'Conv-down':
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_conv_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                 fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
        else:
            raise ValueError('Unsupported module type: ' + k)

    y = torgb(x, y, num_channels=num_channels)
    images_out = y
    assert images_out.dtype == tf.as_dtype(dtype)

    if return_atts:
        with tf.variable_scope('ConcatAtts'):
            atts_out = tf.concat(atts, axis=1)
            return tf.identity(images_out, name='images_out'), tf.identity(atts_out, name='atts_out')
    else:
        return tf.identity(images_out, name='images_out')

