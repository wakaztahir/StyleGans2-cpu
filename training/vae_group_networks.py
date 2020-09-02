#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vae_group_networks.py
# --- Creation Date: 24-08-2020
# --- Last Modified: Wed 02 Sep 2020 02:37:57 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Group VAE networks.
"""
import tensorflow as tf

def build_group_post_E(x, name, scope_idx, group_feats_size, latent_size, is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        flat_x = tf.layers.flatten(x)
        e5 = tf.layers.dense(flat_x, 512, activation=tf.nn.relu, name="e5")

        # Group feats mapping.
        group_feats = tf.layers.dense(e5, group_feats_size, activation=None, name="group_feats_E")

        e6 = tf.layers.dense(group_feats, 512, activation=tf.nn.relu, name="e6")
        means = tf.layers.dense(e6, latent_size, activation=None, name="means")
        log_var = tf.layers.dense(e6, latent_size, activation=None, name="log_var")
    return means, log_var, group_feats

def build_group_post_E_wc(x, name, scope_idx, group_feats_size,
                          con_latent_size, cat_latent_size,
                          is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        flat_x = tf.layers.flatten(x)
        e5 = tf.layers.dense(flat_x, 512, activation=tf.nn.relu, name="e5")

        # Group feats mapping.
        group_feats = tf.layers.dense(e5, group_feats_size, activation=None, name="group_feats_E")

        e6 = tf.layers.dense(group_feats, 512, activation=tf.nn.relu, name="e6")
        means = tf.layers.dense(e6, con_latent_size, activation=None, name="means")
        log_var = tf.layers.dense(e6, con_latent_size, activation=None, name="log_var")

        cat_logits = tf.layers.dense(e6, cat_latent_size, activation=None, name="cat_logits")
        cat_dist = tf.nn.softmax(cat_logits, axis=1)
    return means, log_var, group_feats, cat_dist

def build_group_prior_G(latents_in, name, scope_idx, group_feats_size, is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d1 = tf.layers.dense(latents_in, 512, activation=tf.nn.relu)

        # Group feats mapping.
        group_feats = tf.layers.dense(d1, group_feats_size, activation=None, name="group_feats_G")

        d2 = tf.layers.dense(group_feats, 512, activation=tf.nn.relu)
        d3 = tf.layers.dense(d2, 1024, activation=tf.nn.relu)
        d3_reshaped = tf.reshape(d3, shape=[-1, 64, 4, 4])
    return d3_reshaped, group_feats

def build_group_sim_post_E(x, name, scope_idx, group_feats_size, latent_size, is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        flat_x = tf.layers.flatten(x)
        e5 = tf.layers.dense(flat_x, 256, activation=tf.nn.relu, name="e5")

        # Group feats mapping.
        group_feats = tf.layers.dense(e5, group_feats_size, activation=None, name="group_feats_E")

        means = tf.layers.dense(group_feats, latent_size, activation=None, name="means")
        log_var = tf.layers.dense(group_feats, latent_size, activation=None, name="log_var")
    return means, log_var, group_feats

def build_group_sim_post_up_E(x, name, scope_idx, group_feats_size, latent_size, is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        flat_x = tf.layers.flatten(x)

        # Group feats mapping.
        group_feats = tf.layers.dense(flat_x, group_feats_size, activation=None, name="group_feats_E")

        e5 = tf.layers.dense(group_feats, 256, activation=tf.nn.relu, name="e5")
        means = tf.layers.dense(e5, latent_size, activation=None, name="means")
        log_var = tf.layers.dense(e5, latent_size, activation=None, name="log_var")
    return means, log_var, group_feats

def build_group_sim_post_E_wc(x, name, scope_idx, group_feats_size,
                              con_latent_size, cat_latent_size,
                              is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        flat_x = tf.layers.flatten(x)
        e5 = tf.layers.dense(flat_x, 256, activation=tf.nn.relu, name="e5")

        # Group feats mapping.
        group_feats = tf.layers.dense(e5, group_feats_size, activation=None, name="group_feats_E")

        means = tf.layers.dense(group_feats, con_latent_size, activation=None, name="means")
        log_var = tf.layers.dense(group_feats, con_latent_size, activation=None, name="log_var")

        cat_logits = tf.layers.dense(group_feats, cat_latent_size, activation=None, name="cat_logits")
        cat_dist = tf.nn.softmax(cat_logits, axis=1)
    return means, log_var, group_feats, cat_dist

def build_group_sim_prior_G(latents_in, name, scope_idx, group_feats_size, is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        # Group feats mapping.
        group_feats = tf.layers.dense(latents_in, group_feats_size, activation=None)

        d1 = tf.layers.dense(group_feats, 256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, group_feats

def build_group_sim_prior_down_G(latents_in, name, scope_idx, group_feats_size, is_validation=False):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        d1 = tf.layers.dense(latents_in, 256, activation=tf.nn.relu)

        # Group feats mapping.
        group_feats = tf.layers.dense(d1, group_feats_size, activation=None)

        d2 = tf.layers.dense(group_feats, 1024, activation=tf.nn.relu)
        d2_reshaped = tf.reshape(d2, shape=[-1, 64, 4, 4])
    return d2_reshaped, group_feats
