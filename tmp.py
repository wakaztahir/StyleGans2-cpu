#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: tmp.py
# --- Creation Date: 05-12-2020
# --- Last Modified: Sat 10 Apr 2021 18:03:59 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""
import numpy as np
import tensorflow as tf

def construct_skew_mat(mat_dim, var_idx):
    idx_sum = 0
    for i_k, k in enumerate(range(mat_dim - 1, -1, -1)):
        if idx_sum + k > var_idx:
            break
        else:
            idx_sum += k
    j = mat_dim - k + (var_idx - idx_sum)
    i = mat_dim * i_k + j
    skew_mat_1 = np.zeros(shape=(1, mat_dim * mat_dim), dtype=np.float32)
    skew_mat_1[0, i] = 1.
    skew_mat_1 = np.reshape(skew_mat_1, (1, mat_dim, mat_dim))
    skew_mat = skew_mat_1 - np.transpose(skew_mat_1, (0, 2, 1))
    return skew_mat

def create_split_mask(seq_len, hy_ncut):
    # x: [b, dim]
    # if hy_ncut == 0:
        # return tf.ones(shape=[seq_len, seq_len], dtype=tf.float32)

    split_idx = tf.random.uniform(shape=[hy_ncut],
                                  minval=1,
                                  maxval=seq_len + 1,
                                  dtype=tf.int32)
    split_idx = tf.sort(split_idx, axis=-1)
    idx_range = tf.range(seq_len)
    masks_mul = tf.zeros(shape=[seq_len, seq_len], dtype=tf.float32)
    pre_mask_last = tf.zeros(shape=[seq_len], dtype=tf.float32)
    # for i in range(hy_ncut):
        # pre_mask_tmp = tf.cast(idx_range < split_idx[i:i + 1], tf.float32)
        # mask_tmp = pre_mask_tmp - pre_mask_last
        # mask_mul_tmp = tf.matmul(mask_tmp[:, np.newaxis], mask_tmp[np.newaxis, :])
        # masks_mul = masks_mul + mask_mul_tmp
        # pre_mask_last = pre_mask_tmp

    def cond(i, masks):
        return tf.less(i, hy_ncut)

    def bod(i, masks):
        masks_mul, pre_mask_last = masks
        pre_mask_tmp = tf.cast(idx_range < split_idx[i:i + 1], tf.float32)
        mask_tmp = pre_mask_tmp - pre_mask_last
        mask_mul_tmp = tf.matmul(mask_tmp[:, np.newaxis], mask_tmp[np.newaxis, :])
        masks_mul = masks_mul + mask_mul_tmp
        pre_mask_last = pre_mask_tmp
        i += 1
        return (i, (masks_mul, pre_mask_last))

    i_masks = (0, (masks_mul, pre_mask_last))
    i_final, masks_final = tf.while_loop(cond, bod, i_masks)
    masks_mul, pre_mask_last = masks_final

    def f1():
        return tf.ones(shape=[seq_len], dtype=tf.float32)
    def f2():
        return (1. - tf.cast(idx_range < split_idx[-1:], tf.float32))
    mask_tmp = tf.cond(tf.equal(hy_ncut, 0), f1, f2)
    mask_mul_tmp = tf.matmul(mask_tmp[:, np.newaxis], mask_tmp[np.newaxis, :])
    masks_mul = masks_mul + mask_mul_tmp
    # uniq, _ = tf.unique(split_idx)
    # return masks_mul, split_idx, uniq
    return masks_mul, split_idx
