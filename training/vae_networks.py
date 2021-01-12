#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_networks.py
# --- Creation Date: 14-08-2020
# --- Last Modified: Tue 12 Jan 2021 15:26:09 AEDT
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
from training.vae_standard_networks import build_6layer_conv_G_64
from training.vae_standard_networks import build_standard_conv_G_128
from training.vae_standard_networks import build_standard_prior_G
from training.vae_standard_networks import build_standard_fc_D_64
from training.vae_standard_networks import build_standard_fc_D_128
from training.vae_standard_networks import build_standard_fc_sindis_D_64
from training.vae_standard_networks import build_simple_fc_sindis_D_64
from training.vae_standard_networks import build_fain_conv_G_64
from training.vae_group_networks import build_group_post_E
from training.vae_group_networks import build_group_post_E_wc
from training.vae_group_networks import build_group_prior_G
from training.vae_group_networks import build_group_sim_post_E
from training.vae_group_networks import build_group_sim_post_up_E
from training.vae_group_networks import build_group_sim_post_E_wc
from training.vae_group_networks import build_group_sim_prior_G
from training.vae_group_networks import build_group_sim_prior_G_wc
from training.vae_group_networks import build_group_sim_prior_down_G
from training.vae_group_networks_v2 import build_group_act_sim_prior_G
from training.vae_group_networks_v2 import build_group_act_spl_sim_prior_G
from training.vae_group_networks_v3 import build_group_norm_prior_G
from training.vae_group_networks_v4 import build_group_subspace_prior_G, build_group_subspace_post_E
from training.vae_lie_networks import build_lie_sim_prior_G
from training.vae_lie_networks import build_lie_sim_prior_G_oth
from training.vae_lie_networks import build_lie_sim_prior_G_oth_l2
from training.vae_lie_networks import build_lie_sim_prior_G_oth_nogroup
from training.vae_lie_networks import build_lie_sim_prior_G_oth_squash
from training.vae_so_networks import build_so_prior_G
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
        subgroup_sizes_ls=None,
        subspace_sizes_ls=None,
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    '''
    Modularized VAE encoder.
    '''
    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

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
            x = build_standard_conv_E_64(reals_in=x,
                                         name=k,
                                         scope_idx=scope_idx,
                                         is_validation=is_validation)
        elif k == 'Standard_E_128':
            x = build_standard_conv_E_128(reals_in=x,
                                          name=k,
                                          scope_idx=scope_idx,
                                          is_validation=is_validation)
        elif k == 'Standard_post_E':
            x = build_standard_post_E(x=x,
                                      name=k,
                                      scope_idx=scope_idx,
                                      latent_size=latent_size,
                                      is_validation=is_validation)
            break
        elif k == 'Standard_post_norelu_E':
            x = build_standard_post_E(x=x,
                                      name=k,
                                      scope_idx=scope_idx,
                                      latent_size=latent_size,
                                      use_relu=False,
                                      is_validation=is_validation)
            break
        elif k == 'Group_post_E':
            x = build_group_post_E(x=x,
                                   name=k,
                                   scope_idx=scope_idx,
                                   group_feats_size=group_feats_size,
                                   latent_size=latent_size,
                                   is_validation=is_validation)
            break
        elif k == 'Group_post_E_wc':
            x = build_group_post_E_wc(x=x,
                                      name=k,
                                      scope_idx=scope_idx,
                                      group_feats_size=group_feats_size,
                                      con_latent_size=latent_size,
                                      cat_latent_size=n_discrete,
                                      is_validation=is_validation)
            break
        elif k == 'Group_post_sim_E':
            x = build_group_sim_post_E(x=x,
                                       name=k,
                                       scope_idx=scope_idx,
                                       group_feats_size=group_feats_size,
                                       latent_size=latent_size,
                                       is_validation=is_validation)
            break
        elif k == 'Group_post_sim_up_E':
            x = build_group_sim_post_up_E(x=x,
                                          name=k,
                                          scope_idx=scope_idx,
                                          group_feats_size=group_feats_size,
                                          latent_size=latent_size,
                                          is_validation=is_validation)
            break
        elif k == 'Group_post_sim_E_wc':
            x = build_group_sim_post_E_wc(x=x,
                                          name=k,
                                          scope_idx=scope_idx,
                                          group_feats_size=group_feats_size,
                                          con_latent_size=latent_size,
                                          cat_latent_size=n_discrete,
                                          is_validation=is_validation)
            break
        elif k == 'SBS_post_E':
            x = build_group_subspace_post_E(x=x,
                                            name=k,
                                            scope_idx=scope_idx,
                                            subgroup_sizes_ls=subgroup_sizes_ls,
                                            subspace_sizes_ls=subspace_sizes_ls,
                                            latent_size=latent_size,
                                            is_validation=False)
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
        n_act_points=10,
        lie_alg_init_type='oth',
        lie_alg_init_scale=0.1,
        R_view_scale=1,
        group_feat_type='concat',
        fmap_min=16,
        fmap_max=512,
        fmap_decay=0.15,
        latent_size=10,
        label_size=0,
        module_G_list=None,
        nf_scale=1,
        fmap_base=8,
        mapping_after_exp=False,
        use_sphere_points=False,
        use_learnable_sphere_points=False,
        n_sphere_points=100,
        group_feats_size=400,  # Should be square of an integer.
        hy_ncut=1,
        normalize_alg=True,
        use_alg_var=True,
        subgroup_sizes_ls=None,
        subspace_sizes_ls=None,
        lie_alg_init_type_ls=None,
        forward_eg=False,
        forward_eg_prob=0.3333,
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    '''
    Modularized VAE encoder.
    '''
    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    # Validate arguments.
    assert not is_training or not is_validation

    # Primary inputs.
    latents_in.set_shape(input_shape)
    latents_in = tf.cast(latents_in, dtype)
    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)

    if forward_eg:
        len_gfeats = sum(subgroup_sizes_ls)
        group_feats_E = latents_in[:, -len_gfeats:]
        latents_in = latents_in[:, :-len_gfeats]
    else:
        group_feats_E = None

    # Generator network.
    key_ls, size_ls, count_dlatent_size = split_module_names(module_G_list)
    x = latents_in
    group_feats = tf.zeros([1], dtype=latents_in.dtype)
    group_feats_cat_mat = tf.zeros([1], dtype=latents_in.dtype)
    group_feats_con_mat = tf.zeros([1], dtype=latents_in.dtype)
    lie_alg_feats = tf.zeros([1], dtype=latents_in.dtype)
    lie_alg_basis = tf.zeros([1], dtype=latents_in.dtype)
    act_points = tf.zeros([1], dtype=latents_in.dtype)
    lie_vars = tf.zeros([1], dtype=latents_in.dtype)
    for scope_idx, k in enumerate(key_ls):
        if k == 'Standard_prior_G':
            x, group_feats = \
                build_standard_prior_G(latents_in=x, name=k, scope_idx=scope_idx,
                                       is_validation=is_validation)
        elif k == 'Standard_prior_norelu_G':
            x, group_feats = \
                build_standard_prior_G(latents_in=x, name=k, scope_idx=scope_idx,
                                       use_relu=False, is_validation=is_validation)
        elif k == 'COMA_G':
            x = build_fain_conv_G_64(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                output_shape=[num_channels, resolution, resolution],
                recons_type=recons_type,
                is_validation=is_validation)
        elif k == 'Group_prior_G':
            x, group_feats = build_group_prior_G(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                is_validation=is_validation)
        elif k == 'Group_prior_sim_G':
            x, group_feats = build_group_sim_prior_G(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                is_validation=is_validation)
        elif k == 'Group_prior_sim_G_wc':
            # return d2_reshaped, group_feats, group_feats_cat_mat, group_feats_con_mat
            x, group_feats, group_feats_cat_mat, group_feats_con_mat = build_group_sim_prior_G_wc(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                con_latent_size=latent_size,
                cat_latent_size=n_discrete,
                is_validation=is_validation)
        elif k == 'Group_prior_sim_down_G':
            x, group_feats = build_group_sim_prior_down_G(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                is_validation=is_validation)
        elif k == 'Group_act_prior_sim_G':
            x, group_feats, lie_alg_feats, lie_alg_basis, act_points = build_group_act_sim_prior_G(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                n_act_points=n_act_points,
                lie_alg_init_type=lie_alg_init_type,
                lie_alg_init_scale=lie_alg_init_scale,
                is_validation=is_validation)
        elif k == 'Group_act_spl_prior_sim_G':
            x, group_feats, lie_alg_feats, lie_alg_basis, act_points = build_group_act_spl_sim_prior_G(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                n_act_points=n_act_points,
                lie_alg_init_type=lie_alg_init_type,
                lie_alg_init_scale=lie_alg_init_scale,
                is_validation=is_validation)
        elif k == 'Group_norm_prior_sim_G':
            x, group_feats, lie_alg_feats, lie_alg_basis, lie_vars = build_group_norm_prior_G(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                hy_ncut=hy_ncut,
                lie_alg_init_type=lie_alg_init_type,
                lie_alg_init_scale=lie_alg_init_scale,
                normalize_alg=normalize_alg,
                use_alg_var=use_alg_var,
                is_validation=is_validation)
        elif k == 'Lie_prior_sim_G':
            x, group_feats, lie_alg_feats, lie_alg_basis = build_lie_sim_prior_G(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                lie_alg_init_scale=lie_alg_init_scale,
                is_validation=is_validation)
        elif k == 'Lie_prior_sim_G_oth':
            x, group_feats, lie_alg_feats, lie_alg_basis = build_lie_sim_prior_G_oth(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                lie_alg_init_scale=lie_alg_init_scale,
                is_validation=is_validation)
        elif k == 'Lie_prior_sim_G_oth_l2':
            x, group_feats, lie_alg_feats, lie_alg_basis = build_lie_sim_prior_G_oth_l2(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                lie_alg_init_scale=lie_alg_init_scale,
                is_validation=is_validation)
        elif k == 'Lie_prior_sim_G_oth_nogroup':
            x, group_feats, lie_alg_feats, lie_alg_basis = build_lie_sim_prior_G_oth_nogroup(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                lie_alg_init_scale=lie_alg_init_scale,
                is_validation=is_validation)
        elif k == 'Lie_prior_sim_G_oth_squash':
            x, group_feats, lie_alg_feats, lie_alg_basis = build_lie_sim_prior_G_oth_squash(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                lie_alg_init_scale=lie_alg_init_scale,
                is_validation=is_validation)
        elif k == 'SO_prior_G':
            x, group_feats, lie_alg_feats, lie_alg_basis, lie_vars = build_so_prior_G(
                latents_in=x,
                name=k,
                scope_idx=scope_idx,
                group_feats_size=group_feats_size,
                lie_alg_init_scale=lie_alg_init_scale,
                R_view_scale=R_view_scale,
                group_feat_type=group_feat_type,
                mapping_after_exp=mapping_after_exp,
                use_sphere_points=use_sphere_points,
                use_learnable_sphere_points=use_learnable_sphere_points,
                n_sphere_points=n_sphere_points,
                is_validation=is_validation)
        elif k == 'SBS_prior_G':
            x, group_feats, lie_alg_feats, lie_alg_basis, lie_vars = build_group_subspace_prior_G(
                latents_in=x,
                group_feats_E=group_feats_E,
                name=k,
                scope_idx=scope_idx,
                subgroup_sizes_ls=subgroup_sizes_ls,
                subspace_sizes_ls=subspace_sizes_ls,
                lie_alg_init_type_ls=lie_alg_init_type_ls,
                hy_ncut=hy_ncut,
                lie_alg_init_scale=lie_alg_init_scale,
                normalize_alg=normalize_alg,
                use_alg_var=use_alg_var,
                forward_eg=forward_eg,
                forward_eg_prob=forward_eg_prob,
                is_validation=is_validation)
        elif k == 'Standard_G_64':
            x = build_standard_conv_G_64(
                d2_reshaped=x,
                name=k,
                scope_idx=scope_idx,
                output_shape=[num_channels, resolution, resolution],
                recons_type=recons_type,
                is_validation=is_validation)
            break
        elif k == '6layer_G_64':
            x = build_6layer_conv_G_64(
                d2_reshaped=x,
                name=k,
                scope_idx=scope_idx,
                output_shape=[num_channels, resolution, resolution],
                recons_type=recons_type,
                is_validation=is_validation)
            break
        elif k == 'Standard_G_128':
            x = build_standard_conv_G_128(
                x,
                name=k,
                scope_idx=scope_idx,
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
    return x, group_feats, group_feats_cat_mat, group_feats_con_mat, \
        lie_alg_feats, lie_alg_basis, act_points, lie_vars


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
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

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
            logits, probs = build_standard_fc_D_64(latents=x,
                                                   name=k,
                                                   scope_idx=scope_idx)
            break
        elif k == 'Standard_D_128':
            logits, probs = build_standard_fc_D_128(latents=x,
                                                    name=k,
                                                    scope_idx=scope_idx)
            break
        elif k == 'Standard_D_sindis_64':
            logits = build_standard_fc_sindis_D_64(latents=x,
                                                   name=k,
                                                   scope_idx=scope_idx)
            probs = logits
            break
        elif k == 'Simple_D_sindis_64':
            logits = build_simple_fc_sindis_D_64(latents=x,
                                                 name=k,
                                                 scope_idx=scope_idx)
            probs = logits
            break
        else:
            raise ValueError('Not supported module key:', k)

    # Return requested outputs.
    logits = tf.identity(logits, name='discrim_logits')
    probs = tf.identity(probs, name='discrim_probs')
    return logits, probs
