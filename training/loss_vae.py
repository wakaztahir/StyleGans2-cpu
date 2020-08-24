#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vae.py
# --- Creation Date: 15-08-2020
# --- Last Modified: Mon 24 Aug 2020 17:13:20 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss of VAEs. Some code borrowed from disentanglement_lib.
"""
import numpy as np
import math
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
from training.utils import get_return_v

def sample_from_latent_distribution(z_mean, z_logvar):
    """Samples from the Gaussian distribution defined by z_mean and z_logvar."""
    return tf.add(z_mean,
                  tf.exp(z_logvar / 2) *
                  tf.random_normal(tf.shape(z_mean), 0, 1),
                  name="sampled_latent_variable")


def make_reconstruction_loss(true_images, reconstructed_images, recons_type='l2_loss'):
    """Wrapper that creates reconstruction loss."""
    with tf.variable_scope("reconstruction_loss"):
        if recons_type == 'l2_loss':
            loss = tf.reduce_sum(tf.square(
                true_images - reconstructed_images), [1, 2, 3])
            # loss = tf.reduce_mean(tf.square(
                # true_images - reconstructed_images), [1, 2, 3])
        else:
            flattened_dim = np.prod(true_images.get_shape().as_list()[1:])
            reconstructed_images = tf.reshape(
                reconstructed_images, shape=[-1, flattened_dim])
            true_images = tf.reshape(true_images, shape=[-1, flattened_dim])
            # true_images = (true_images + 1.) / 2. # drange_net has been set to [0, 1]
            loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=reconstructed_images, labels=true_images),
                axis=1)
    return loss


def compute_gaussian_kl(z_mean, z_logvar):
    """Compute KL divergence between input Gaussian and Standard Normal."""
    with tf.variable_scope("kl_loss"):
        return 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) -
                                   z_logvar - 1, [1])
        # return tf.reduce_mean(
            # 0.5 * tf.reduce_sum(
                # tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]),
            # name="kl_loss")


def shuffle_codes(z):
    """Shuffles latent variables across the batch.
    Args:
        z: [batch_size, num_latent] representation.
    Returns:
        shuffled: [batch_size, num_latent] shuffled representation across the batch.
    """
    z_shuffle = []
    for i in range(z.get_shape()[1]):
        z_shuffle.append(tf.random_shuffle(z[:, i]))
    shuffled = tf.stack(z_shuffle, 1, name="latent_shuffled")
    return shuffled

def compute_covariance_z_mean(z_mean):
    """Computes the covariance of z_mean.
       Uses cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T.
       Args:
         z_mean: Encoder mean, tensor of size [batch_size, num_latent].
       Returns:
         cov_z_mean: Covariance of encoder mean, tensor of size [num_latent,
         num_latent].
    """
    expectation_z_mean_z_mean_t = tf.reduce_mean(
        tf.expand_dims(z_mean, 2) * tf.expand_dims(z_mean, 1), axis=0)
    expectation_z_mean = tf.reduce_mean(z_mean, axis=0)
    cov_z_mean = tf.subtract(
        expectation_z_mean_z_mean_t,
        tf.expand_dims(expectation_z_mean, 1) * tf.expand_dims(
            expectation_z_mean, 0))
    return cov_z_mean

def regularize_diag_off_diag_dip(covariance_matrix, lambda_od, lambda_d):
    """Compute on and off diagonal regularizers for DIP-VAE models.
       Penalize deviations of covariance_matrix from the identity matrix. Uses
       different weights for the deviations of the diagonal and off diagonal entries.
       Args:
         covariance_matrix: Tensor of size [num_latent, num_latent] to regularize.
         lambda_od: Weight of penalty for off diagonal elements.
         lambda_d: Weight of penalty for diagonal elements.
       Returns:
       dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
    """
    covariance_matrix_diagonal = tf.diag_part(covariance_matrix)
    covariance_matrix_off_diagonal = covariance_matrix - tf.diag(
        covariance_matrix_diagonal)
    dip_regularizer = tf.add(
        lambda_od * tf.reduce_sum(covariance_matrix_off_diagonal**2),
        lambda_d * tf.reduce_sum((covariance_matrix_diagonal - 1)**2))
    return dip_regularizer

def gaussian_log_density(samples, mean, log_var):
    pi = tf.constant(math.pi)
    normalization = tf.log(2. * pi)
    inv_sigma = tf.exp(-log_var)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

def total_correlation(z, z_mean, z_logvar):
    """Estimate of total correlation on a batch.
       We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
       log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
       for the minimization. The constant should be equal to (num_latents - 1) *
       log(batch_size * dataset_size)
       Args:
         z: [batch_size, num_latents]-tensor with sampled representation.
         z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
         z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
       Returns:
    Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    log_qz_prob = gaussian_log_density(
        tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
        tf.expand_dims(z_logvar, 0))
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = tf.reduce_sum(
        tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
        axis=1,
        keepdims=False)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = tf.reduce_logsumexp(
        tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
        axis=1,
        keepdims=False)
    # return tf.reduce_mean(log_qz - log_qz_product)
    return log_qz - log_qz_product

def beta_vae(E, G, opt, training_set, minibatch_size, reals, labels,
             latent_type='normal', hy_beta=1, recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var = get_return_v(E.get_output_for(reals, labels, is_training=True), 2)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)
    sampled = sample_from_latent_distribution(means, log_var)
    reconstructions = get_return_v(G.get_output_for(sampled, labels, is_training=True), 1)
    reconstruction_loss = make_reconstruction_loss(reals, reconstructions,
                                                   recons_type=recons_type)
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)

    loss = reconstruction_loss + hy_beta * kl_loss
    loss = autosummary('Loss/beta_vae_loss', loss)
    elbo = reconstruction_loss + kl_loss
    elbo = autosummary('Loss/beta_vae_elbo', elbo)
    return loss

def betatc_vae(E, G, opt, training_set, minibatch_size, reals, labels,
             latent_type='normal', hy_beta=1, recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var = get_return_v(E.get_output_for(reals, labels, is_training=True), 2)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)
    sampled = sample_from_latent_distribution(means, log_var)
    reconstructions = get_return_v(G.get_output_for(sampled, labels, is_training=True), 1)
    reconstruction_loss = make_reconstruction_loss(reals, reconstructions,
                                                   recons_type=recons_type)
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)

    tc = (hy_beta - 1.) * total_correlation(sampled, means, log_var)
    # return tc + kl_loss
    elbo = reconstruction_loss + kl_loss
    elbo = autosummary('Loss/betatc_vae_elbo', elbo)
    loss = elbo + tc
    loss = autosummary('Loss/betatc_vae_loss', loss)
    return loss

def dip_vae(E, G, opt, training_set, minibatch_size, reals, labels,
            latent_type='normal', dip_type='dip_vae_i',
            lambda_d_factor=10., lambda_od=1.,
            recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var = get_return_v(E.get_output_for(reals, labels, is_training=True), 2)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)
    sampled = sample_from_latent_distribution(means, log_var)
    reconstructions = get_return_v(G.get_output_for(sampled, labels, is_training=True), 1)
    reconstruction_loss = make_reconstruction_loss(reals, reconstructions,
                                                   recons_type=recons_type)
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)

    # Regularization
    cov_z_mean = compute_covariance_z_mean(means)
    lambda_d = lambda_d_factor * lambda_od
    if dip_type == 'dip_vae_i':
        # mu = means is [batch_size, num_latent]
        # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
        cov_dip_regularizer = regularize_diag_off_diag_dip(
            cov_z_mean, lambda_od, lambda_d)
    elif dip_type == 'dip_vae_ii':
        cov_enc = tf.matrix_diag(tf.exp(log_var))
        expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
        cov_z = expectation_cov_enc + cov_z_mean
        cov_dip_regularizer = regularize_diag_off_diag_dip(
            cov_z, lambda_od, lambda_d)
    else:
        raise NotImplementedError("DIP variant not supported.")

    elbo = reconstruction_loss + kl_loss
    elbo = autosummary('Loss/dip_vae_elbo', elbo)
    loss = elbo + cov_dip_regularizer
    loss = autosummary('Loss/dip_vae_loss', loss)
    return loss

def factor_vae_G(E, G, D, opt, training_set, minibatch_size, reals, labels,
                 latent_type='normal', hy_gamma=1, recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var = get_return_v(E.get_output_for(reals, labels, is_training=True), 2)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)
    sampled = sample_from_latent_distribution(means, log_var)
    reconstructions = get_return_v(G.get_output_for(sampled, labels, is_training=True), 1)

    logits, probs = get_return_v(D.get_output_for(sampled, is_training=True), 2)
    # tc = E[log(p_real)-log(p_fake)] = E[logit_real - logit_fake]
    tc_loss = logits[:, 0] - logits[:, 1]
    # tc_loss = tf.reduce_mean(tc_loss, axis=0)
    reconstruction_loss = make_reconstruction_loss(reals, reconstructions,
                                                   recons_type=recons_type)
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)
    elbo = reconstruction_loss + kl_loss
    elbo = autosummary('Loss/fac_vae_elbo', elbo)
    loss = elbo + hy_gamma * tc_loss
    loss = autosummary('Loss/fac_vae_loss', loss)
    return loss

def factor_vae_D(E, D, opt, training_set, minibatch_size, reals, labels,
                 latent_type='normal'):
    _ = opt, training_set
    means, log_var = get_return_v(E.get_output_for(reals, labels, is_training=True), 2)
    sampled = sample_from_latent_distribution(means, log_var)
    shuffled = shuffle_codes(sampled)
    logits, probs = get_return_v(D.get_output_for(sampled, is_training=True), 2)
    _, probs_shuffled = get_return_v(D.get_output_for(shuffled, is_training=True), 2)
    loss = -(0.5 * tf.log(probs[:, 0]) + 0.5 * tf.log(probs_shuffled[:, 1]))
    # loss = -tf.add(
        # 0.5 * tf.reduce_mean(tf.log(probs[:, 0])),
        # 0.5 * tf.reduce_mean(tf.log(probs_shuffled[:, 1])),
        # name="discriminator_loss")
    loss = autosummary('Loss/fac_vae_discr_loss', loss)
    return loss

def factor_vae_sindis_G(E, G, D, opt, training_set, minibatch_size, reals, labels,
                 latent_type='normal', hy_gamma=1, recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var = get_return_v(E.get_output_for(reals, labels, is_training=True), 2)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)
    sampled = sample_from_latent_distribution(means, log_var)
    reconstructions = get_return_v(G.get_output_for(sampled, labels, is_training=True), 1)

    fake_scores_out, _ = get_return_v(D.get_output_for(sampled, is_training=True), 2)
    tc_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    # loss = tf.reduce_mean(loss)

    reconstruction_loss = make_reconstruction_loss(reals, reconstructions,
                                                   recons_type=recons_type)
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)
    elbo = reconstruction_loss + kl_loss
    elbo = autosummary('Loss/fac_vae_elbo', elbo)
    loss = elbo + hy_gamma * tc_loss
    loss = autosummary('Loss/fac_vae_loss', loss)
    return loss

def factor_vae_sindis_D(E, D, opt, training_set, minibatch_size, reals, labels,
                 latent_type='normal'):
    _ = opt, training_set
    means, log_var = get_return_v(E.get_output_for(reals, labels, is_training=True), 2)
    sampled = sample_from_latent_distribution(means, log_var)
    shuffled = shuffle_codes(sampled)
    real_scores_out, _ = get_return_v(D.get_output_for(shuffled, is_training=True), 2)
    fake_scores_out, _ = get_return_v(D.get_output_for(sampled, is_training=True), 2)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    loss = autosummary('Loss/fac_vae_discr_loss', loss)
    return loss

def make_group_feat_loss(group_feats_E, group_feats_G, minibatch_size,
                         group_loss_type):
    group_feats_G_ori = group_feats_G[:minibatch_size]
    group_feats_G_sum = group_feats_G[minibatch_size:]
    mat_dim = int(math.sqrt(group_feats_E.get_shape().as_list()[1]))
    group_feats_G_mat = tf.reshape(group_feats_G,
                                   [minibatch_size+minibatch_size//2,
                                    mat_dim, mat_dim])
    loss = 0
    if 'rec' in group_loss_type:
        loss += tf.reduce_sum(tf.square(
                group_feats_E - group_feats_G_ori), axis=1)
    if 'mat' in group_loss_type:
        assert mat_dim * mat_dim == group_feats_E.get_shape().as_list()[1]
        group_feats_E_mat = tf.reshape(group_feats_E,
                                       [minibatch_size, mat_dim, mat_dim])
        group_feats_G_sum_mat = tf.reshape(group_feats_G_sum,
                                       [minibatch_size, mat_dim, mat_dim])
        group_feats_E_mul_mat = tf.matmul(group_feats_E_mat[:minibatch_size],
                                          group_feats_E_mat[:minibatch_size])
        loss += tf.reduce_sum(tf.square(
                group_feats_E_mul_mat - group_feats_G_sum_mat), axis=[1, 2])
    if 'oth' in group_loss_type:
        group_feats_G_mat2 = tf.matmul(group_feats_G_mat, group_feats_G_mat,
                                       transpose_b=True)
        G_mat_eye = tf.eye(mat_dim, dtype=group_feats_G_mat2.dtype,
                           batch_shape=[minibatch_size+minibatch_size//2])
        loss += tf.reduce_sum(tf.square(
                group_feats_G_mat2 - G_mat_eye), axis=[1, 2])
    if 'det' in group_loss_type:
        group_feats_G_det = tf.linalg.det(group_feats_G_mat, name='G_mat_det')
        group_feats_one_det = tf.ones(tf.shape(group_feats_G_det),
                                      dtype=group_feats_G_det.dtype)
        loss += tf.reduce_sum(tf.square(
                group_feats_G_det - group_feats_one_det), axis=[1])
    return loss

def group_vae(E, G, opt, training_set, minibatch_size, reals, labels,
              latent_type='normal', hy_beta=1, group_loss_type='rec_mat',
              recons_type='bernoulli_loss'):
    _ = opt, training_set
    means, log_var, group_feats_E = get_return_v(E.get_output_for(reals, labels, is_training=True), 3)
    kl_loss = compute_gaussian_kl(means, log_var)
    kl_loss = autosummary('Loss/kl_loss', kl_loss)

    sampled = sample_from_latent_distribution(means, log_var)
    sampled_sum = sampled[:minibatch_size//2] + sampled[minibatch_size//2:]
    sampled_all = tf.concat([sampled, sampled_sum], axis=0)
    labels_all = tf.concat([labels, labels[:minibatch_size//2]], axis=0)
    reconstructions, group_feats_G = get_return_v(G.get_output_for(sampled, labels, is_training=True), 2)
    group_feat_loss = make_group_feat_loss(group_feats_E, group_feats_G, minibatch_size,
                                           group_loss_type)

    reconstruction_loss = make_reconstruction_loss(reals, reconstructions[:minibatch_size],
                                                   recons_type=recons_type)
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    reconstruction_loss = autosummary('Loss/recons_loss', reconstruction_loss)

    elbo = reconstruction_loss + kl_loss
    elbo = autosummary('Loss/group_vae_elbo', elbo)
    loss = elbo + hy_beta * group_feat_loss
    loss = autosummary('Loss/group_vae_loss', loss)
    return loss
