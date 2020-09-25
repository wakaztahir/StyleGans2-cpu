#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_networks.py
# --- Creation Date: 14-08-2020
# --- Last Modified: Thu 24 Sep 2020 23:38:48 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
VAE networks.
"""
import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
from training.vc_modular_networks2 import split_module_names
from training.vae_standard_networks import build_standard_conv_E_64
from training.vae_standard_networks import build_standard_conv_E_128
from training.vae_standard_networks import build_standard_post_E
from training.vae_standard_networks import build_standard_conv_G_64
from training.vae_standard_networks import build_standard_conv_G_128
from training.vae_standard_networks import build_standard_prior_G
from training.vae_standard_networks import build_standard_fc_D_64
from training.vae_standard_networks import build_standard_fc_D_128
from training.vae_standard_networks import build_standard_fc_sindis_D_64
from training.vae_standard_networks import build_simple_fc_sindis_D_64
from training.vae_group_networks import build_group_post_E
from training.vae_group_networks import build_group_post_E_wc
from training.vae_group_networks import build_group_prior_G
from training.vae_group_networks import build_group_sim_post_E
from training.vae_group_networks import build_group_sim_post_up_E
from training.vae_group_networks import build_group_sim_post_E_wc
from training.vae_group_networks import build_group_sim_prior_G
from training.vae_group_networks import build_group_sim_prior_G_wc
from training.vae_group_networks import build_group_sim_prior_down_G
from training.vae_lie_networks import build_lie_sim_prior_G
from training.vae_lie_networks import build_lie_sim_prior_G_oth
from training.utils import get_return_v

#----------------------------------------------------------------------------
# VAE main Encoder.
def E_main_modular(
        reals_in,  # First input: Real images [minibatch, image_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        input_shape=None,  # Input image shape.
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        dtype='float32',  # Data type to use for activations and outputs.
        n_discrete=0,  # Number of discrete categories.
        fmap_min=16,
        fmap_max=512,
        fmap_decay=0.15,
        latent_size=10,
        label_size=0,
        module_E_list=None,
        nf_scale=1,
        fmap_base=8,
        group_feats_size=400,  # Should be square of an integer.
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    '''
    Modularized VAE encoder.
    '''

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min, fmap_max)

    # Validate arguments.
    assert not is_training or not is_validation

    # Primary inputs.
    reals_in.set_shape(input_shape)
    reals_in = tf.cast(reals_in, dtype)
    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)

    # Encoder network.
    key_ls, size_ls, count_dlatent_size = split_module_names(module_E_list)
    x = reals_in
    for scope_idx, k in enumerate(key_ls):
        if k == 'Standard_E_64':
            x = build_standard_conv_E_64(reals_in=x, name=k, scope_idx=scope_idx,
                                         is_validation=is_validation)
        elif k == 'Standard_E_128':
            x = build_standard_conv_E_128(reals_in=x, name=k, scope_idx=scope_idx,
                                          is_validation=is_validation)
        elif k == 'Standard_post_E':
            x = build_standard_post_E(x=x, name=k, scope_idx=scope_idx,
                                      latent_size=latent_size, is_validation=is_validation)
            break
        elif k == 'Standard_post_norelu_E':
            x = build_standard_post_E(x=x, name=k, scope_idx=scope_idx,
                                      latent_size=latent_size, use_relu=False,
                                      is_validation=is_validation)
            break
        elif k == 'Group_post_E':
            x = build_group_post_E(x=x, name=k, scope_idx=scope_idx,
                                   group_feats_size=group_feats_size,
                                   latent_size=latent_size, is_validation=is_validation)
            break
        elif k == 'Group_post_E_wc':
            x = build_group_post_E_wc(x=x, name=k, scope_idx=scope_idx,
                                      group_feats_size=group_feats_size,
                                      con_latent_size=latent_size,
                                      cat_latent_size=n_discrete,
                                      is_validation=is_validation)
            break
        elif k == 'Group_post_sim_E':
            x = build_group_sim_post_E(x=x, name=k, scope_idx=scope_idx,
                                       group_feats_size=group_feats_size,
                                       latent_size=latent_size, is_validation=is_validation)
            break
        elif k == 'Group_post_sim_up_E':
            x = build_group_sim_post_up_E(x=x, name=k, scope_idx=scope_idx,
                                       group_feats_size=group_feats_size,
                                       latent_size=latent_size, is_validation=is_validation)
            break
        elif k == 'Group_post_sim_E_wc':
            x = build_group_sim_post_E_wc(x=x, name=k, scope_idx=scope_idx,
                                          group_feats_size=group_feats_size,
                                          con_latent_size=latent_size,
                                          cat_latent_size=n_discrete,
                                          is_validation=is_validation)
            break
        else:
            raise ValueError('Not supported module key:', k)

    assert isinstance(x, tuple)
    # if len(x) == 2:
        # means, log_var = x
    # elif len(x) == 3:
        # means, log_var, feats = x
    # else:
        # raise ValueError('Strange return value: len(x) > 3.')

    # # Return requested outputs.
    # means = tf.identity(means, name='means')
    # log_var = tf.identity(log_var, name='log_var')
    # if is_validation:
        # return means
    # else:
        # return means, log_var
    return x

#----------------------------------------------------------------------------
# VAE main Generator.
def G_main_modular(
        latents_in,  # First input: Real images [minibatch, image_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        input_shape=None,  # Latent code shape.
        num_channels=3,  # Number of channels in images.
        resolution=64,  # Resolution of images.
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        dtype='float32',  # Data type to use for activations and outputs.
        n_discrete=0,  # Number of discrete categories.
        recons_type='bernoulli_loss',  # Reconstruction type.
        fmap_min=16,
        fmap_max=512,
        fmap_decay=0.15,
        latent_size=10,
        label_size=0,
        module_G_list=None,
        nf_scale=1,
        fmap_base=8,
        group_feats_size=400,  # Should be square of an integer.
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    '''
    Modularized VAE encoder.
    '''

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min, fmap_max)

    # Validate arguments.
    assert not is_training or not is_validation

    # Primary inputs.
    latents_in.set_shape(input_shape)
    latents_in = tf.cast(latents_in, dtype)
    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)

    # Generator network.
    key_ls, size_ls, count_dlatent_size = split_module_names(module_G_list)
    x = latents_in
    group_feats = None
    group_feats_cat_mat = tf.zeros([1], dtype=latents_in.dtype)
    group_feats_con_mat = tf.zeros([1], dtype=latents_in.dtype)
    lie_alg_feats = tf.zeros([1], dtype=latents_in.dtype)
    lie_alg_basis = tf.zeros([1], dtype=latents_in.dtype)
    for scope_idx, k in enumerate(key_ls):
        if k == 'Standard_prior_G':
            x, group_feats = \
                build_standard_prior_G(latents_in=x, name=k, scope_idx=scope_idx,
                                       is_validation=is_validation)
        elif k == 'Standard_prior_norelu_G':
            x, group_feats = \
                build_standard_prior_G(latents_in=x, name=k, scope_idx=scope_idx,
                                       use_relu=False, is_validation=is_validation)
        elif k == 'Group_prior_G':
            x, group_feats = build_group_prior_G(latents_in=x, name=k, scope_idx=scope_idx,
                                                 group_feats_size=group_feats_size,
                                                 is_validation=is_validation)
        elif k == 'Group_prior_sim_G':
            x, group_feats = build_group_sim_prior_G(latents_in=x, name=k, scope_idx=scope_idx,
                                                     group_feats_size=group_feats_size,
                                                     is_validation=is_validation)
        elif k == 'Group_prior_sim_G_wc':
            # return d2_reshaped, group_feats, group_feats_cat_mat, group_feats_con_mat
            x, group_feats, group_feats_cat_mat, group_feats_con_mat = build_group_sim_prior_G_wc(
                latents_in=x, name=k, scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                con_latent_size=latent_size,
                cat_latent_size=n_discrete,
                is_validation=is_validation)
        elif k == 'Group_prior_sim_down_G':
            x, group_feats = build_group_sim_prior_down_G(latents_in=x, name=k, scope_idx=scope_idx,
                                                     group_feats_size=group_feats_size,
                                                     is_validation=is_validation)
        elif k == 'Lie_prior_sim_G':
            x, group_feats, lie_alg_feats, lie_alg_basis = build_lie_sim_prior_G(latents_in=x, name=k, scope_idx=scope_idx,
                                                                  group_feats_size=group_feats_size,
                                                                  is_validation=is_validation)
        elif k == 'Lie_prior_sim_G_oth':
            x, group_feats, lie_alg_feats, lie_alg_basis = build_lie_sim_prior_G_oth(latents_in=x, name=k, scope_idx=scope_idx,
                                                                  group_feats_size=group_feats_size,
                                                                  is_validation=is_validation)
        elif k == 'Standard_G_64':
            x = build_standard_conv_G_64(d2_reshaped=x, name=k, scope_idx=scope_idx,
                                         output_shape=[num_channels, resolution, resolution],
                                         recons_type=recons_type,
                                         is_validation=is_validation)
            break
        elif k == 'Standard_G_128':
            x = build_standard_conv_G_128(x, name=k, scope_idx=scope_idx,
                                          output_shape=[num_channels, resolution, resolution],
                                          recons_type=recons_type,
                                          is_validation=is_validation)
            break
        else:
            raise ValueError('Not supported module key:', k)

    # Return requested outputs.
    # x = tf.identity(x, name='fake_x')
    # if group_feats is not None:
        # return x, group_feats
    # else:
        # return x
    return x, group_feats, group_feats_cat_mat, group_feats_con_mat, lie_alg_feats, lie_alg_basis

#----------------------------------------------------------------------------
# Factor-VAE main Discriminator.
def D_factor_vae_modular(
        latents_in,  # First input: Real images [minibatch, image_size].
        input_shape=None,  # Latent code shape.
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        dtype='float32',  # Data type to use for activations and outputs.
        fmap_min=16,
        fmap_max=512,
        fmap_decay=0.15,
        latent_size=10,
        label_size=0,
        module_D_list=None,
        nf_scale=1,
        fmap_base=8,
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    '''
    Modularized Factor-VAE discriminator.
    '''

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min, fmap_max)

    # Validate arguments.
    assert not is_training or not is_validation

    # Primary inputs.
    latents_in.set_shape(input_shape)
    latents_in = tf.cast(latents_in, dtype)

    # Discriminator network.
    key_ls, size_ls, count_dlatent_size = split_module_names(module_D_list)
    x = latents_in
    for scope_idx, k in enumerate(key_ls):
        if k == 'Standard_D_64':
            logits, probs = build_standard_fc_D_64(latents=x, name=k, scope_idx=scope_idx)
            break
        elif k == 'Standard_D_128':
            logits, probs = build_standard_fc_D_128(latents=x, name=k, scope_idx=scope_idx)
            break
        elif k == 'Standard_D_sindis_64':
            logits = build_standard_fc_sindis_D_64(latents=x, name=k, scope_idx=scope_idx)
            probs = logits
            break
        elif k == 'Simple_D_sindis_64':
            logits = build_simple_fc_sindis_D_64(latents=x, name=k, scope_idx=scope_idx)
            probs = logits
            break
        else:
            raise ValueError('Not supported module key:', k)

    # Return requested outputs.
    logits = tf.identity(logits, name='discrim_logits')
    probs = tf.identity(probs, name='discrim_probs')
    return logits, probs
