#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vid_networks.py
# --- Creation Date: 24-03-2020
# --- Last Modified: Tue 31 Mar 2020 18:11:22 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Vid Networks
"""

import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d
from dnnlib.tflib.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act
from training.networks_stylegan2 import get_weight, dense_layer, conv2d_layer
from training.networks_stylegan2 import apply_bias_act, naive_upsample_2d
from training.networks_stylegan2 import naive_downsample_2d, modulated_conv2d_layer
from training.networks_stylegan2 import minibatch_stddev_layer
from training.spatial_biased_extended_networks import torgb, get_conditional_modifier
from training.spatial_biased_extended_networks import get_att_heat
from training.spatial_biased_modular_networks import split_module_names, build_D_layers
from training.spatial_biased_modular_networks import build_C_global_layers
from training.spatial_biased_modular_networks import build_local_heat_layers, build_local_hfeat_layers
from training.spatial_biased_modular_networks import build_noise_layer, build_conv_layer
from stn.stn import spatial_transformer_network as transformer


#----------------------------------------------------------------------------
# Vid main Generator
def G_main_vid(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        return_dlatents=False,  # Return dlatents in addition to the images?
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        components=dnnlib.EasyDict(
        ),  # Container for sub-networks. Retained between calls.
        mapping_func='G_mapping_vid',  # Build func name for the mapping network.
        synthesis_func='G_synthesis_vid_modular',  # Build func name for the synthesis network.
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    # Validate arguments.
    assert not is_training or not is_validation

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network(
            'G_vid_synthesis', func_name=globals()[synthesis_func], **kwargs)
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_vid_mapping',
                                           func_name=globals()[mapping_func],
                                           dlatent_broadcast=None,
                                           **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in,
                                                 labels_in,
                                                 is_training=is_training,
                                                 **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Evaluate synthesis network.
    deps = []
    if 'lod' in components.synthesis.vars:
        deps.append(tf.assign(components.synthesis.vars['lod'], lod_in))
    with tf.control_dependencies(deps):
        images_out = components.synthesis.get_output_for(
            dlatents,
            is_training=is_training,
            force_clean_graph=is_template_graph,
            **kwargs)

    # Return requested outputs.
    images_out = tf.identity(images_out, name='images_out')
    if return_dlatents:
        return images_out, dlatents
    return images_out


def G_mapping_vid(
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

    with tf.variable_scope('LabelConcat'):
        x = tf.concat([labels_in, x], axis=1)

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')


def G_synthesis_vid_modular(
        dlatents_withl_in,  # Input: Disentangled latents (W) [minibatch, label_size+dlatent_size].
        dlatent_size=7,  # Disentangled latent (W) dimensionality. Including discrete info, rotation, scaling, xy shearing, and xy translation.
        label_size=0,  # Label dimensionality, 0 if no labels.
        module_list=None,  # A list containing module names, which represent semantic latents (exclude labels).
        num_channels=1,  # Number of output color channels.
        resolution=64,  # Output resolution.
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
        single_const=True,  # If only use a single constant feature at the begining.
        **_kwargs):  # Ignore unrecognized keyword args.
    '''
    Modularized vid generator network.
    '''
    print('fmap_decay: ', fmap_decay)
    resolution_log2 = int(np.log2(resolution))  # == 6 for resolution 64
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    num_layers = resolution_log2 * 2 - 2  # == 10 for resolution 64

    act = nonlinearity
    images_out = None

    # Note that module_list may include modules not containing latents,
    # e.g. Conv layers (size in this case means number of conv layers).
    key_ls, size_ls, count_dlatent_size, n_content = split_module_names(
        module_list)
    print('In key_ls:', key_ls)
    print('In size_ls:', size_ls)
    print('In count_dlatent_size:', count_dlatent_size)
    if label_size > 0:
        key_ls.insert(0, 'Label')
        size_ls.insert(0, label_size)
        n_content += label_size
    # module_dict = collections.OrderedDict(zip(key_ls, size_ls))

    # Primary inputs.
    assert dlatent_size == count_dlatent_size
    dlatents_withl_in.set_shape([None, label_size + count_dlatent_size])
    dlatents_withl_in = tf.cast(dlatents_withl_in, dtype)

    # Early layers consists of 4x4 constant layer.
    y = None
    if single_const:
        with tf.variable_scope('4x4'):
            with tf.variable_scope('Const'):
                x = tf.get_variable(
                    'const',
                    shape=[1, 128, 4, 4],
                    initializer=tf.initializers.random_normal())
                x = tf.tile(tf.cast(x, dtype),
                            [tf.shape(dlatents_withl_in)[0], 1, 1, 1])
    else:
        with tf.variable_scope('4x4'):
            with tf.variable_scope('Const'):
                x = tf.get_variable(
                    'const',
                    shape=[n_content, 128, 4, 4],
                    initializer=tf.initializers.random_normal())

    subkwargs = EasyDict()
    subkwargs.update(dlatents_withl_in=dlatents_withl_in,
                     n_content=n_content,
                     act=act,
                     dtype=dtype,
                     resample_kernel=resample_kernel,
                     fused_modconv=fused_modconv,
                     use_noise=use_noise,
                     randomize_noise=randomize_noise)

    # Build modules by module_dict.
    start_idx = 0
    # print('module_dict:', module_dict)
    # for scope_idx, k in enumerate(module_dict):
    for scope_idx, k in enumerate(key_ls):
        if (k.startswith('Label')) or (k.startswith('D_global')):
            # e.g. {'Label': 3}, {'D_global': 3}
            x = build_D_layers(x,
                               name=k,
                               n_latents=size_ls[scope_idx],
                               start_idx=start_idx,
                               scope_idx=scope_idx,
                               single_const=single_const,
                               fmaps=nf(scope_idx),
                               **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_global'):
            # e.g. {'C_global': 2}
            x = build_C_global_layers(x,
                                      name=k,
                                      n_latents=size_ls[scope_idx],
                                      start_idx=start_idx,
                                      scope_idx=scope_idx,
                                      **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_nocond_global'):
            # e.g. {'C_nocond_global': 2}
            x = build_C_global_nocond_layers(x,
                                      name=k,
                                      n_latents=size_ls[scope_idx],
                                      start_idx=start_idx,
                                      scope_idx=scope_idx,
                                      fmaps=nf(scope_idx),
                                      **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_local_heat'):
            # e.g. {'C_local_heat': 4}
            x = build_local_heat_layers(x,
                                        name=k,
                                        n_latents=size_ls[scope_idx],
                                        start_idx=start_idx,
                                        scope_idx=scope_idx,
                                        fmaps=nf(scope_idx),
                                        **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_local_hfeat'):
            # e.g. {'C_local_hfeat_size': 4}
            x = build_local_hfeat_layers(x,
                                         name=k,
                                         n_latents=size_ls[scope_idx],
                                         start_idx=start_idx,
                                         scope_idx=scope_idx,
                                         fmaps=nf(scope_idx),
                                         **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('Noise'):
            # e.g. {'Noise': 1}
            x = build_noise_layer(x,
                                  name=k,
                                  n_layers=size_ls[scope_idx],
                                  scope_idx=scope_idx,
                                  fmaps=nf(scope_idx),
                                  **subkwargs)
        elif k.startswith('Conv'):
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_conv_layer(x,
                                 name=k,
                                 n_layers=size_ls[scope_idx],
                                 scope_idx=scope_idx,
                                 fmaps=nf(scope_idx),
                                 **subkwargs)
        else:
            raise ValueError('Unsupported module type: ' + k)

    y = torgb(x, y, num_channels=num_channels)
    images_out = y
    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out,
                       name='images_out')


def conv3d_layer(x, fmaps, kernel, down=False, resample_kernel=None, 
                 gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([3, kernel, kernel, x.shape[1].value, fmaps],
                   gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    x = tf.nn.conv3d(x, tf.cast(w, x.dtype), data_format='NCDHW', strides=[1,1,1,1,1], padding='SAME')
    if down:
        x = tf.nn.avg_pool3d(x, ksize=[1, 2, 2], strides=[1, 2, 2], padding='VALID', data_format='NCDHW')
    return x

def downsample_3d(y):
    with tf.variable_scope('Downsample'):
        y = tf.nn.avg_pool3d(y, ksize=[1, 2, 2], strides=[1, 2, 2], padding='VALID', data_format='NCDHW')
    return y

def apply_bias_act_3d(x, act='linear', alpha=None, gain=None, lrmul=1, bias_var='bias'):
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    x = tf.nn.bias_add(x, b, data_format='NCDHW')
    if act == 'linear':
        x = x
    elif act == 'lrelu':
        if alpha is None:
            alpha = 0.2
        x = tf.nn.leaky_relu(x, alpha)
    elif act == 'relu':
        x = tf.nn.relu(x)
    return x

#----------------------------------------------------------------------------
# Vid Head network.

def vid_head(
        fake_in,  # First input: generated image from z [minibatch, channel, n_frames, height, width].
        C_delta_idxes,  # Second input: the index of the varied latent.
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        D_global_size=0,
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    fake_in.set_shape([None, num_channels, None, resolution, resolution])
    fake_in = tf.cast(fake_in, dtype)
    C_delta_idxes.set_shape([None, dlatent_size])
    C_delta_idxes = tf.cast(C_delta_idxes, dtype)

    vid_in = fake_in

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = conv3d_layer(y, fmaps=nf(res - 1), kernel=1)
            t = apply_bias_act_3d(t, act=act)
        return t if x is None else x + t

    # def block(x, res):  # res = 2..resolution_log2
        # t = x
        # with tf.variable_scope('Conv0'):
            # x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3),
                               # act=act)
        # with tf.variable_scope('Conv1_down'):
            # x = apply_bias_act(conv2d_layer(x,
                                            # fmaps=nf(res - 2),
                                            # kernel=3,
                                            # down=True,
                                            # resample_kernel=resample_kernel),
                               # act=act)
        # if architecture == 'resnet':
            # with tf.variable_scope('Skip'):
                # t = conv2d_layer(t,
                                 # fmaps=nf(res - 2),
                                 # kernel=1,
                                 # down=True,
                                 # resample_kernel=resample_kernel)
                # x = (x + t) * (1 / np.sqrt(2))
        # return x

    def block(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('Conv3D_0'):
            x = conv3d_layer(x, fmaps=nf(res - 1), kernel=3)
            x = apply_bias_act_3d(x, act=act)
        with tf.variable_scope('Conv1_down'):
            x = conv3d_layer(x, fmaps=nf(res - 2), kernel=3, down=True)
            x = apply_bias_act_3d(x, act=act)
        return x

    # def downsample(y):
        # with tf.variable_scope('Downsample'):
            # return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = vid_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('I_%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample_3d(y)

    # Final layers.
    with tf.variable_scope('I_4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        with tf.variable_scope('Conv'):
            x = conv3d_layer(x, fmaps=nf(1), kernel=3)
            x = apply_bias_act_3d(x, act=act)
        with tf.variable_scope('Global_temporal_pool'):
            x = tf.reduce_mean(x, axis=2)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=64), act=act)

    print('before from C_delta_idxes, x.get_shape:', x.get_shape().as_list())
    print('before from C_delta_idxes, x.shape:', x.shape)
    print('before from C_delta_idxes, C_delta_idxes.shape:', C_delta_idxes.shape)
    # From C_delta_idxes
    with tf.variable_scope('I_From_C_Delta_Idx'):
        x_from_delta = apply_bias_act(dense_layer(C_delta_idxes, fmaps=32), act=act)
        x = tf.concat([x, x_from_delta], axis=1)

    # For MINE
    with tf.variable_scope('I_Output'):
        with tf.variable_scope('Dense_T_0'):
            x = apply_bias_act(dense_layer(x, fmaps=128), act=act)
        with tf.variable_scope('Dense_T_1'):
            x = apply_bias_act(dense_layer(x, fmaps=1))

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x

#----------------------------------------------------------------------------
# Vid Cluster Head network.

def vid_naive_cluster_head(
        fake_in,  # First input: generated image from z [minibatch, channel, n_frames, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        D_global_size=0,
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    fake_in.set_shape([None, num_channels, None, resolution, resolution])
    fake_in = tf.cast(fake_in, dtype)

    vid_in = fake_in

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = conv3d_layer(y, fmaps=nf(res - 1), kernel=1)
            t = apply_bias_act_3d(t, act=act)
        return t if x is None else x + t

    def block(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('Conv3D_0'):
            x = conv3d_layer(x, fmaps=nf(res - 1), kernel=3)
            x = apply_bias_act_3d(x, act=act)
        with tf.variable_scope('Conv1_down'):
            x = conv3d_layer(x, fmaps=nf(res - 2), kernel=3, down=True)
            x = apply_bias_act_3d(x, act=act)
        return x

    # Main layers.
    x = None
    y = vid_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('I_%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample_3d(y)

    # Final layers.
    with tf.variable_scope('I_4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        with tf.variable_scope('Conv'):
            x = conv3d_layer(x, fmaps=nf(1), kernel=3)
            x = apply_bias_act_3d(x, act=act)
        with tf.variable_scope('Global_temporal_pool'):
            x = tf.reduce_mean(x, axis=2)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output.
    with tf.variable_scope('I_Output'):
        with tf.variable_scope('Dense_VC'):
            x = apply_bias_act(
                dense_layer(x,
                            fmaps=dlatent_size - D_global_size))

    assert x.dtype == tf.as_dtype(dtype)
    return x


def build_C_global_nocond_layers(x,
                                 name,
                                 n_latents,
                                 start_idx,
                                 scope_idx,
                                 dlatents_withl_in,
                                 act,
                                 fused_modconv,
                                 fmaps=128,
                                 **kwargs):
    '''
    Build continuous latent layers, e.g. C_global layers.
    '''
    with tf.variable_scope(name + '-' + str(scope_idx)):
        with tf.variable_scope('Conv0'):
            C_global_latents = apply_bias_act(dense_layer(
                dlatents_withl_in[:, start_idx:start_idx + n_latents], fmaps=128),
                                  act=act)
        # C_global_latents = dlatents_withl_in[:, start_idx:start_idx +
                                             # n_latents]
        with tf.variable_scope('Modulate'):
            x = apply_bias_act(modulated_conv2d_layer(x,
                                                      C_global_latents,
                                                      fmaps=fmaps,
                                                      kernel=3,
                                                      up=False,
                                                      fused_modconv=fused_modconv),
                               act=act)
    return x
