#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training_loop_gan.py
# --- Creation Date: 11-12-2020
# --- Last Modified: Fri 11 Dec 2020 18:49:15 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Training loop file for GAN related networks.
Code borrowed from training_loop.py of NVIDIA.
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
from training.training_loop import process_reals
from training.utils import add_outline, get_grid_latents, get_return_v
from training.utils import training_schedule

#----------------------------------------------------------------------------
# Main training script.

def training_loop_gan(
        G_args={},  # Options for generator network.
        E_args={},  # Options for encoder network.
        D_args={},  # Options for discriminator network.
        G_opt_args={},  # Options for generator optimizer.
        D_opt_args={},  # Options for discriminator optimizer.
        G_loss_args={},  # Options for generator loss.
        D_loss_args={},  # Options for discriminator loss.
        dataset_args={},  # Options for dataset.load_dataset().
        sched_args={},  # Options for train.TrainingSchedule.
        grid_args={},  # Options for train.setup_snapshot_image_grid().
        metric_arg_list=[],  # Options for MetricGroup.
        tf_config={},  # Options for tflib.init_tf().
        data_dir=None,  # Directory to load datasets from.
        G_smoothing_kimg=10.0,  # Half-life of the running average of generator weights.
        minibatch_repeats=1,  # Number of minibatches to run before adjusting training parameters.
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        mirror_augment=False,  # Enable mirror augment?
        drange_net=[-1, 1],  # Dynamic range used when feeding image data to the networks.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = only save 'networks-final.pkl'.
        save_tf_graph=False,  # Include full TensorFlow computation graph in the tfevents file?
        save_weight_histograms=False,  # Include weight histograms in the tfevents file?
        resume_pkl=None,  # Network pickle to resume training from, None = train from scratch.
        resume_kimg=0.0,  # Assumed training progress at the beginning. Affects reporting and training schedule.
        resume_time=0.0,  # Assumed wallclock time at the beginning. Affects reporting.
        resume_with_new_nets=False,  # Construct new networks according to G_args and D_args before resuming training?
        traversal_grid=False,  # Used for disentangled representation learning.
        n_discrete=0,  # Number of discrete latents in model.
        n_continuous=4,  # Number of continuous latents in model.
        avg_mv_for_E=False,  # If use average moving for I.
        topk_dims_to_show=20, # Number of top disentant dimensions to show in a snapshot.
        n_samples_per=10):  # Number of samples for each line in traversal.

    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # If use Discriminator.
    use_E = E_args is not None

    # Load training set.
    training_set = dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir),
                                        verbose=True,
                                        **dataset_args)
    grid_size, grid_reals, grid_labels = misc.setup_snapshot_image_grid(
        training_set, **grid_args)
    grid_fakes = add_outline(grid_reals, width=1)
    misc.save_image_grid(grid_reals,
                         dnnlib.make_run_dir_path('reals.png'),
                         drange=training_set.dynamic_range,
                         grid_size=grid_size)

    # Construct or load networks.
    with tf.device('/gpu:0'):
        if resume_pkl is None or resume_with_new_nets:
            print('Constructing networks...')
            D = tflib.Network('D',
                              num_channels=training_set.shape[0],
                              resolution=training_set.shape[1],
                              label_size=training_set.label_size,
                              input_shape=[None]+training_set.shape,
                              **D_args)
            G = tflib.Network('G',
                              num_channels=training_set.shape[0],
                              resolution=training_set.shape[1],
                              label_size=training_set.label_size,
                              input_shape=[None, n_discrete+G_args.latent_size],
                              **G_args)
            if use_E:
                E = tflib.Network('E',
                                  num_channels=training_set.shape[0],
                                  resolution=training_set.shape[1],
                                  label_size=training_set.label_size,
                                  input_shape=[None]+training_set.shape,
                                  **E_args)
                if avg_mv_for_E:
                    Es = E.clone('Es')
            Gs = G.clone('Gs')
        if resume_pkl is not None:
            print('Loading networks from "%s"...' % resume_pkl)
            if use_E:
                if avg_mv_for_E:
                    rG, rD, rE, rGs, rEs = misc.load_pkl(resume_pkl)
                else:
                    rG, rD, rE, rGs = misc.load_pkl(resume_pkl)
            else:
                rG, rD, rGs = misc.load_pkl(resume_pkl)

            if resume_with_new_nets:
                D.copy_vars_from(rD)
                G.copy_vars_from(rG)
                if use_E:
                    E.copy_vars_from(rE)
                    if avg_mv_for_E:
                        Es.copy_vars_from(rEs)
                Gs.copy_vars_from(rGs)
            else:
                D = rD
                G = rG
                if use_E:
                    E = rE
                    if avg_mv_for_E:
                        Es = rEs
                Gs = rGs

    # Print layers and generate initial image snapshot.
    D.print_layers()
    G.print_layers()
    if use_E:
        E.print_layers()
    sched = training_schedule(cur_nimg=total_kimg * 1000,
                              training_set=training_set,
                              **sched_args)
    if traversal_grid:
        topk_dims = np.arange(min(topk_dims_to_show, n_continuous))
        grid_size, grid_latents, grid_labels = get_grid_latents(
            n_discrete, n_continuous, n_samples_per, G, grid_labels, topk_dims)
    else:
        grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])
    print('grid_size:', grid_size)
    print('grid_latents.shape:', grid_latents.shape)
    print('grid_labels.shape:', grid_labels.shape)
    grid_fakes, _, _, _, _, _, _, lie_vars = get_return_v(Gs.run(grid_latents,
                                                                 grid_labels,
                                                                 is_validation=True,
                                                                 minibatch_size=sched.minibatch_gpu,
                                                                 randomize_noise=True), 8)
    print('Lie_vars:', lie_vars[0])
    grid_fakes = add_outline(grid_fakes, width=1)
    misc.save_image_grid(grid_fakes,
                         dnnlib.make_run_dir_path('fakes_init.png'),
                         drange=drange_net,
                         grid_size=grid_size)

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
        Gs_beta = 0.5**tf.div(tf.cast(minibatch_size_in,
                                      tf.float32), G_smoothing_kimg *
                              1000.0) if G_smoothing_kimg > 0.0 else 0.0

    # Setup optimizers.
    G_opt_args = dict(G_opt_args)
    G_opt_args['minibatch_multiplier'] = minibatch_multiplier
    G_opt_args['learning_rate'] = lrate_in
    G_opt = tflib.Optimizer(name='TrainG', **G_opt_args)
    D_opt_args = dict(D_opt_args)
    D_opt_args['minibatch_multiplier'] = minibatch_multiplier
    D_opt_args['learning_rate'] = lrate_in
    D_opt = tflib.Optimizer(name='TrainD', **D_opt_args)

    # Build training graph for each GPU.
    data_fetch_ops = []
    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):

            # Create GPU-specific shadow copies of G and D.
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            if use_E:
                E_gpu = E if gpu == 0 else E.clone(E.name + '_shadow')
            else:
                E_gpu = None

            # Fetch training data via temporary variables.
            with tf.name_scope('DataFetch'):
                sched = training_schedule(cur_nimg=int(resume_kimg * 1000),
                                          training_set=training_set,
                                          **sched_args)
                reals_var = tf.Variable(name='reals', trainable=False,
                                        initial_value=tf.zeros([sched.minibatch_gpu] +
                                                               training_set.shape))
                labels_var = tf.Variable(name='labels',
                                         trainable=False,
                                         initial_value=tf.zeros([
                                             sched.minibatch_gpu,
                                             training_set.label_size
                                         ]))
                reals_write, labels_write = training_set.get_minibatch_tf()
                reals_write, labels_write = process_reals(
                    reals_write, labels_write, 0., mirror_augment,
                    training_set.dynamic_range, drange_net)
                reals_write = tf.concat(
                    [reals_write, reals_var[minibatch_gpu_in:]], axis=0)
                labels_write = tf.concat(
                    [labels_write, labels_var[minibatch_gpu_in:]], axis=0)
                data_fetch_ops += [tf.assign(reals_var, reals_write)]
                data_fetch_ops += [tf.assign(labels_var, labels_write)]
                reals_read = reals_var[:minibatch_gpu_in]
                labels_read = labels_var[:minibatch_gpu_in]

            # Evaluate loss functions.
            with tf.name_scope('G_loss'):
                G_loss, G_reg = dnnlib.util.call_func_by_name(
                    G=G_gpu, D=D_gpu, E=E_gpu, opt=G_opt,
                    training_set=training_set,
                    minibatch_size=minibatch_gpu_in,
                    reals=reals_read, labels=labels_read, **G_loss_args)
            with tf.name_scope('D_loss'):
                D_loss, D_reg = dnnlib.util.call_func_by_name(
                    G=G_gpu, D=D_gpu, E=E_gpu, opt=D_opt,
                    training_set=training_set,
                    minibatch_size=minibatch_gpu_in,
                    reals=reals_read, labels=labels_read, **D_loss_args)
            if G_reg is not None: G_loss += G_reg
            if D_reg is not None: D_loss += D_reg

            # Register gradients.
            if use_E:
                gpu_trainables = collections.OrderedDict(
                        list(E_gpu.trainables.items()) +
                        list(G_gpu.trainables.items()))
            else:
                gpu_trainables = G_gpu.trainables
            G_opt.register_gradients(tf.reduce_mean(G_loss),
                                     gpu_trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss),
                                     D_gpu.trainables)

    # Setup training ops.
    data_fetch_op = tf.group(*data_fetch_ops)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)
    if avg_mv_for_E:
        Es_update_op = Es.setup_as_moving_average_of(E, beta=Gs_beta)

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
        G.setup_weight_histograms()
        D.setup_weight_histograms()
        if use_E:
            E.setup_weight_histograms()
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
        sched = training_schedule(cur_nimg=cur_nimg,
                                  training_set=training_set,
                                  **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        training_set.configure(sched.minibatch_gpu, 0)

        # Run training ops.
        feed_dict = {
            lrate_in: sched.G_lrate,
            minibatch_size_in: sched.minibatch_size,
            minibatch_gpu_in: sched.minibatch_gpu
        }
        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched.minibatch_size,
                           sched.minibatch_gpu * num_gpus)
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1

            # Fast path without gradient accumulation.
            if len(rounds) == 1:
                tflib.run([G_train_op, data_fetch_op], feed_dict)
                if avg_mv_for_E:
                    tflib.run([D_train_op, Gs_update_op, Es_update_op], feed_dict)
                else:
                    tflib.run([D_train_op, Gs_update_op], feed_dict)

            # Slow path with gradient accumulation.
            else:
                for _round in rounds:
                    tflib.run(G_train_op, feed_dict)
                if avg_mv_for_E:
                    tflib.run([Gs_update_op, Es_update_op], feed_dict)
                else:
                    tflib.run(Gs_update_op, feed_dict)
                for _round in rounds:
                    tflib.run(data_fetch_op, feed_dict)
                    tflib.run(D_train_op, feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
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
                   autosummary('Progress/minibatch', sched.minibatch_size),
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
            if network_snapshot_ticks is not None and (
                    cur_tick % network_snapshot_ticks == 0 or done):
                pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' %
                                               (cur_nimg // 1000))
                if use_E:
                    if avg_mv_for_E:
                        misc.save_pkl((G, D, E, Gs, Es), pkl)
                    else:
                        misc.save_pkl((G, D, E, Gs), pkl)
                else:
                    misc.save_pkl((G, D, Gs), pkl)
                met_outs = metrics.run(pkl,
                                       run_dir=dnnlib.make_run_dir_path(),
                                       data_dir=dnnlib.convert_path(data_dir),
                                       num_gpus=num_gpus,
                                       tf_config=tf_config,
                                       is_vae=True, use_E=use_E,
                                       Gs_kwargs=dict(is_validation=True))
                if 'tpl_per_dim' in met_outs:
                    avg_distance_per_dim = met_outs['tpl_per_dim'] # shape: (n_continuous)
                    topk_dims = np.argsort(avg_distance_per_dim)[::-1][:topk_dims_to_show] # shape: (20)
                else:
                    topk_dims = np.arange(min(topk_dims_to_show, n_continuous))

            if image_snapshot_ticks is not None and (
                    cur_tick % image_snapshot_ticks == 0 or done):
                if traversal_grid:
                    grid_size, grid_latents, grid_labels = get_grid_latents(
                        n_discrete, n_continuous, n_samples_per, G, grid_labels, topk_dims)
                else:
                    grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])

                grid_fakes, _, _, _, _, _, _, lie_vars = get_return_v(Gs.run(grid_latents,
                                                                             grid_labels,
                                                                             is_validation=True,
                                                                             minibatch_size=sched.minibatch_gpu,
                                                                             randomize_noise=True), 8)
                print('Lie_vars:', lie_vars[0])
                grid_fakes = add_outline(grid_fakes, width=1)
                misc.save_image_grid(grid_fakes,
                                     dnnlib.make_run_dir_path(
                                         'fakes%06d.png' % (cur_nimg // 1000)),
                                     drange=drange_net,
                                     grid_size=grid_size)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            dnnlib.RunContext.get().update('%.2f' % 0,
                                           cur_epoch=cur_nimg // 1000,
                                           max_epoch=total_kimg)
            maintenance_time = dnnlib.RunContext.get(
            ).get_last_update_interval() - tick_time

    # Save final snapshot.
    if use_E:
        if avg_mv_for_E:
            misc.save_pkl((G, D, E, Gs, Es),
                          dnnlib.make_run_dir_path('network-final.pkl'))
        else:
            misc.save_pkl((G, D, E, Gs),
                          dnnlib.make_run_dir_path('network-final.pkl'))
    else:
        misc.save_pkl((G, D, Gs),
                      dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()
    training_set.close()


#----------------------------------------------------------------------------
