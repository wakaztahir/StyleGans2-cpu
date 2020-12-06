#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: tmp.py
# --- Creation Date: 05-12-2020
# --- Last Modified: Sat 05 Dec 2020 21:03:33 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""
import numpy as np


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
