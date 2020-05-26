#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training_loop_infernet.py
# --- Creation Date: 26-05-2020
# --- Last Modified: Tue 26 May 2020 20:59:11 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Training an inference network for a generator.
"""

import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
from metrics import metric_base
from training.training_loop import process_reals, training_schedule
from training.training_loop_dsp import get_grid_latents

def add_outline(images, width=1):
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]
    for i in range(num):
        images[i, :, 0:width, :] = 255
        images[i, :, -width:, :] = 255
        images[i, :, :, 0:width] = 255
        images[i, :, :, -width:] = 255
    return images

#----------------------------------------------------------------------------
# Main training script.


def training_loop_infernet(
        I_args={},  # Options for infogan-head/vcgan-head network.
        I_opt_args={},  # Options for discriminator optimizer.
        loss_args={},  # Options for discriminator loss.
        sched_args={},  # Options for train.TrainingSchedule.
        grid_args={},  # Options for train.setup_snapshot_image_grid().
        metric_arg_list=[],  # Options for MetricGroup.
        tf_config={},  # Options for tflib.init_tf().
        minibatch_repeats=4,  # Number of minibatches to run before adjusting training parameters.
        lazy_regularization=True,  # Perform regularization as a separate training step?
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        mirror_augment=False,  # Enable mirror augment?
        drange_net=[
            -1, 1
        ],  # Dynamic range used when feeding image data to the networks.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = only save 'networks-final.pkl'.
        save_tf_graph=False,  # Include full TensorFlow computation graph in the tfevents file?
        save_weight_histograms=False,  # Include weight histograms in the tfevents file?
        G_pkl=None,  # The G to load.
        resume_pkl=None,  # Network pickle to resume training from, None = train from scratch.
        resume_kimg=0.0,  # Assumed training progress at the beginning. Affects reporting and training schedule.
        resume_time=0.0,  # Assumed wallclock time at the beginning. Affects reporting.
        resume_with_new_nets=False,  # Construct new networks according to G_args and D_args before resuming training?
        n_samples_per=10):  # Number of samples for each line in traversal.

    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # Construct or load networks.
    with tf.device('/gpu:0'):
        G, D, I, Gs = misc.load_pkl(G_pkl)
        if resume_pkl is None or resume_with_new_nets:
            print('Constructing networks...')
            I = tflib.Network('I',
                              num_channels=Gs.output_shapes[1],
                              resolution=Gs.output_shapes[2],
                              **I_args)
        if resume_pkl is not None:
            print('Loading networks from "%s"...' % resume_pkl)
            rI, rGs = misc.load_pkl(resume_pkl)
            if resume_with_new_nets:
                I.copy_vars_from(rI)
                Gs.copy_vars_from(rGs)
            else:
                I = rI
                Gs = rGs

    # Print layers and generate initial image snapshot.
    Gs.print_layers()
    I.print_layers()

    # Setup training inputs.
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_size_in = tf.placeholder(tf.int32,
                                           name='minibatch_size_in',
                                           shape=[])
        minibatch_gpu_in = tf.placeholder(tf.int32,
                                          name='minibatch_gpu_in',
                                          shape=[])
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in *
                                                     num_gpus)

    # Setup optimizers.
    I_opt_args = dict(I_opt_args)
    I_opt_args['minibatch_multiplier'] = minibatch_multiplier
    I_opt_args['learning_rate'] = lrate_in
    I_opt = tflib.Optimizer(name='TrainI', **I_opt_args)

    # Build training graph for each GPU.
    data_fetch_ops = []
    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):

            # Create GPU-specific shadow copies of G and D.
            I_gpu = I if gpu == 0 else I.clone(I.name + '_shadow')
            G_gpu = Gs if gpu == 0 else Gs.clone(Gs.name + '_shadow')

            # Evaluate loss functions.
            with tf.name_scope('I_loss'):
                loss, reg = dnnlib.util.call_func_by_name(
                    G=G_gpu, I=I_gpu, opt=I_opt,
                    minibatch_size=minibatch_gpu_in,
                    **loss_args)

            # Register gradients.
            if reg is not None: loss += reg

            I_opt.register_gradients(tf.reduce_mean(loss), I_gpu.trainables)

    # Setup training ops.
    I_train_op = I_opt.apply_updates()

    # Finalize graph.
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)
    tflib.init_uninitialized_vars()

    print('Initializing logs...')
    summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        I.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('',
                                   cur_epoch=resume_kimg,
                                   max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    running_mb_counter = 0
    while cur_nimg < total_kimg * 1000:
        if dnnlib.RunContext.get().should_stop(): break

        # Choose training parameters and configure training ops.
        assert sched_args.minibatch_size % (sched_args.minibatch_gpu * num_gpus) == 0

        # Run training ops.
        feed_dict = {
            lrate_in: sched_args.lrate,
            minibatch_size_in: sched_args.minibatch_size,
            minibatch_gpu_in: sched_args.minibatch_gpu
        }
        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched_args.minibatch_size,
                           sched_args.minibatch_gpu * num_gpus)
            cur_nimg += sched_args.minibatch_size
            running_mb_counter += 1

            # Fast path without gradient accumulation.
            if len(rounds) == 1:
                tflib.run([I_train_op], feed_dict)
            # Slow path with gradient accumulation.
            else:
                for _round in rounds:
                    tflib.run(I_train_op, feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched_args.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()
            total_time = dnnlib.RunContext.get().get_time_since_start(
            ) + resume_time

            # Report progress.
            print(
                'tick %-5d kimg %-8.1f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f'
                % (autosummary('Progress/tick', cur_tick),
                   autosummary('Progress/kimg', cur_nimg / 1000.0),
                   autosummary('Progress/minibatch', sched_args.minibatch_size),
                   dnnlib.util.format_time(
                       autosummary('Timing/total_sec', total_time)),
                   autosummary('Timing/sec_per_tick', tick_time),
                   autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                   autosummary('Timing/maintenance_sec', maintenance_time),
                   autosummary('Resources/peak_gpu_mem_gb',
                               peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if network_snapshot_ticks is not None and (cur_tick % network_snapshot_ticks == 0 or done):
                pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((I, G), pkl)
                metrics.run(pkl, run_dir=dnnlib.make_run_dir_path(), num_gpus=num_gpus, tf_config=tf_config)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            dnnlib.RunContext.get().update('%.2f' % sched_args.lod,
                                           cur_epoch=cur_nimg // 1000,
                                           max_epoch=total_kimg)
            maintenance_time = dnnlib.RunContext.get(
            ).get_last_update_interval() - tick_time

    # Save final snapshot.
    misc.save_pkl((I, G), dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()


#----------------------------------------------------------------------------
