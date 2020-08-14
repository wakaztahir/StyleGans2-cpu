#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vae.py
# --- Creation Date: 15-08-2020
# --- Last Modified: Sat 15 Aug 2020 02:37:05 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss of VAEs. Some code borrowed from disentanglement_lib.
"""
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary


def sample_from_latent_distribution(z_mean, z_logvar):
    """Samples from the Gaussian distribution defined by z_mean and z_logvar."""
    return tf.add(z_mean,
                  tf.exp(z_logvar / 2) *
                  tf.random_normal(tf.shape(z_mean), 0, 1),
                  name="sampled_latent_variable")


def make_reconstruction_loss(true_images, reconstructed_images):
    """Wrapper that creates reconstruction loss."""
    with tf.variable_scope("reconstruction_loss"):
        per_sample_loss = tf.reduce_sum(tf.square(
            true_images - tf.nn.sigmoid(reconstructed_images)), [1, 2, 3])
    return per_sample_loss


def compute_gaussian_kl(z_mean, z_logvar):
    """Compute KL divergence between input Gaussian and Standard Normal."""
    with tf.variable_scope("kl_loss"):
        return 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1])


def beta_vae(E, G, opt, training_set, minibatch_size, reals, labels,
             latent_type='normal', hy_beta=1):
    _ = opt, training_set
    means, log_var = E.get_output_for(reals, labels, is_training=True)
    kl_loss = compute_gaussian_kl(means, log_var)
    sampled = sample_from_latent_distribution(means, log_var)
    reconstructions = G.get_output_for(sampled, labels, is_training=True)
    reconstruction_loss = make_reconstruction_loss(reals, reconstructions)
    loss = reconstruction_loss + hy_beta * kl_loss
    loss = autosummary('Loss/beta_vae_loss', loss)
    elbo = reconstruction_loss + kl_loss
    elbo = autosummary('Loss/beta_vae_elbo', elbo)
    return loss
