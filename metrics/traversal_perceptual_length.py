#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: traversal_perceptual_length.py
# --- Creation Date: 12-05-2020
# --- Last Modified: Wed 13 May 2020 02:15:02 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Traversal Perceptual Length (TPL)."""

import numpy as np
import pdb
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from metrics.perceptual_path_length import normalize, slerp
from training import misc

#----------------------------------------------------------------------------

class TPL(metric_base.MetricBase):
    def __init__(self, n_samples_per_dim, minibatch_per_gpu, crop, Gs_overrides, **kwargs):
        super().__init__(**kwargs)
        self.crop = crop
        self.minibatch_per_gpu = minibatch_per_gpu
        self.Gs_overrides = Gs_overrides
        self.n_samples_per_dim = n_samples_per_dim

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        Gs_kwargs = dict(Gs_kwargs)
        Gs_kwargs.update(self.Gs_overrides)
        minibatch_size = num_gpus * self.minibatch_per_gpu

        # Construct TensorFlow graph.
        n_continuous = Gs.input_shape[1]
        distance_expr = []
        eval_dim_phs = []
        lat_start_alpha_phs = []
        lat_end_alpha_phs = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                noise_vars = [var for name, var in Gs_clone.components.synthesis.vars.items() if name.startswith('noise')]

                # Latent pairs placeholder
                eval_dim = tf.placeholder(tf.int32)
                lat_start_alpha = tf.placeholder(tf.float32) # should be in [0, 1]
                lat_end_alpha = tf.placeholder(tf.float32) # should be in [0, 1]
                eval_dim_phs.append(eval_dim)
                lat_start_alpha_phs.append(lat_start_alpha)
                lat_end_alpha_phs.append(lat_end_alpha)
                eval_dim_mask = tf.tile(tf.one_hot(eval_dim, n_continuous)[tf.newaxis, :] > 0, [self.minibatch_per_gpu, 1])
                lerp_t = tf.linspace(lat_start_alpha, lat_end_alpha, self.minibatch_per_gpu) # [b]

                lat_t0 = tf.zeros([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                lat_t0_min2 = lat_t0 - 2
                lat_t0 = tf.where(eval_dim_mask, lat_t0_min2, lat_t0) # [b, n_continuous]

                lat_t1 = tf.zeros([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                lat_t1_add2 = lat_t1 + 2
                lat_t1 = tf.where(eval_dim_mask, lat_t1_add2, lat_t1) # [b, n_continuous]
                lat_e = tflib.lerp(lat_t0, lat_t1, lerp_t[:, tf.newaxis]) # [b, n_continuous]

                labels = tf.reshape(self._get_random_labels_tf(self.minibatch_per_gpu), [self.minibatch_per_gpu, -1])
                dlat_e = Gs_clone.components.mapping.get_output_for(lat_e, labels, **Gs_kwargs)

                # Synthesize images.
                with tf.control_dependencies([var.initializer for var in noise_vars]): # use same noise inputs for the entire minibatch
                    images = Gs_clone.components.synthesis.get_output_for(dlat_e, randomize_noise=False, **Gs_kwargs)
                    images = tf.cast(images, tf.float32)

                # Crop only the face region.
                if self.crop:
                    c = int(images.shape[2] // 8)
                    images = images[:, :, c*3 : c*7, c*2 : c*6]

                # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
                factor = images.shape[2] // 256
                if factor > 1:
                    images = tf.reshape(images, [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
                    images = tf.reduce_mean(images, axis=[3,5])

                # Scale dynamic range from [-1,1] to [0,255] for VGG.
                images = (images + 1) * (255 / 2)

                # Evaluate perceptual distance.
                img_e0 = images[:-1]
                img_e1 = images[1:]
                distance_measure = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl')
                distance_tmp = distance_measure.get_output_for(img_e0, img_e1)
                print('distance_tmp.shape:', distance_tmp.get_shape().as_list())
                distance_expr.append(distance_tmp)

        # Sampling loop
        all_distances = []
        sum_distances = []
        n_segs_per_dim = (self.n_samples_per_dim - 1) // ((self.minibatch_per_gpu - 1) * num_gpus)
        self.n_samples_per_dim = n_segs_per_dim * ((self.minibatch_per_gpu - 1) * num_gpus) + 1
        alphas = np.linspace(0., 1., num=(n_segs_per_dim * num_gpus)+1)
        for i in range(n_continuous):
            self._report_progress(i, n_continuous)
            dim_distances = []
            for j in range(n_segs_per_dim):
                fd = {}
                for k_gpu in range(num_gpus):
                    fd.update({eval_dim_phs[k_gpu]:i,
                               lat_start_alpha_phs[k_gpu]:alphas[j*num_gpus+k_gpu],
                               lat_end_alpha_phs[k_gpu]:alphas[j*num_gpus+k_gpu+1]})
                dim_distances += tflib.run(distance_expr, feed_dict=fd)
            dim_distances = np.concatenate(dim_distances, axis=0)
            print('dim_distances.shape:', dim_distances.shape)
            all_distances.append(dim_distances)
            sum_distances.append(np.sum(dim_distances))
        print('sum_distances for each dim:', sum_distances)
        self._report_result(np.mean(sum_distances))
        # pdb.set_trace()

#----------------------------------------------------------------------------
