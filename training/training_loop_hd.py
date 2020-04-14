#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training_loop_hd.py
# --- Creation Date: 07-04-2020
# --- Last Modified: Tue 14 Apr 2020 15:57:23 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
HD disentanglement main training script.
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
from training.training_loop_dsp import get_grid_latents
from training.training_loop import process_reals

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set_resolution_log2,
    lod_initial_resolution  = None,     # Image resolution used at the beginning.
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    minibatch_size_base     = 32,       # Global minibatch size.
    minibatch_size_dict     = {},       # Resolution-specific overrides.
    minibatch_gpu_base      = 4,        # Number of samples processed at a time by one GPU.
    minibatch_gpu_dict      = {},       # Resolution-specific overrides.
    I_lrate_base            = 0.002,    # Learning rate for the generator.
    I_lrate_dict            = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 4,        # Default interval of progress snapshots.
    tick_kimg_dict          = {8:28, 16:24, 32:20, 64:16, 128:12, 256:8, 512:6, 1024:4}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    if lod_initial_resolution is None:
        s.lod = 0.0
    else:
        s.lod = training_set_resolution_log2
        s.lod -= np.floor(np.log2(lod_initial_resolution))
        s.lod -= phase_idx
        if lod_transition_kimg > 0:
            s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set_resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch_size = minibatch_size_dict.get(s.resolution, minibatch_size_base)
    s.minibatch_gpu = minibatch_gpu_dict.get(s.resolution, minibatch_gpu_base)

    # Learning rate.
    s.I_lrate = I_lrate_dict.get(s.resolution, I_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.I_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

def print_traj(prior_traj_latents_show):
    # prior_traj_latents_show  shape: [batch, n_samples_per, n_continuous])
    print(prior_traj_latents_show[0,0,::10])
    print(np.max(prior_traj_latents_show[0,0]))
    print(np.min(prior_traj_latents_show[0,0]))
    for i in range(prior_traj_latents_show.shape[0]):
        line = []
        for j in range(prior_traj_latents_show.shape[1]):
            feat = prior_traj_latents_show[i, j]
            norm = np.sqrt(np.sum(feat * feat))
            line.append(norm)
        print(line)
    return

#----------------------------------------------------------------------------
# Main training script.

def training_loop_hd(
    I_args                  = {},       # Options for generator network.
    M_args                  = {},       # Options for discriminator network.
    I_info_args             = {},       # Options for class network.
    I_opt_args              = {},       # Options for generator optimizer.
    I_loss_args             = {},       # Options for generator loss.
    resume_G_pkl            = None,     # G network pickle to help training.
    dataset_args            = {},       # Options for dataset.load_dataset().
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    data_dir                = None,     # Directory to load datasets from.
    I_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    lazy_regularization     = True,     # Perform regularization as a separate training step?
    I_reg_interval          = 4,        # How often the perform regularization for G? Ignored if lazy_regularization=False.
    reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,    # Enable mirror augment?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = only save 'networks-final.pkl'.
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    resume_pkl              = None,     # Network pickle to resume training from, None = train from scratch.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0,      # Assumed wallclock time at the beginning. Affects reporting.
    resume_with_new_nets    = False,    # Construct new networks according to I_args and M_args before resuming training?
    traversal_grid          = False,    # Used for disentangled representation learning.
    n_discrete=0,  # Number of discrete latents in model.
    n_continuous=10,  # Number of continuous latents in model.
    n_samples_per=4,  # Number of samples for each line in traversal.
    use_hd_with_cls=False,  # If use info_loss.
    resolution_manual=1024,  # Resolution of generated images.
    use_level_training=False,  # If use level training (hierarchical optimization strategy).
    level_I_kimg=1000,  # Number of kimg of tick for I_level training.
    use_std_in_m=False,  # If output prior std in M net.
    prior_latent_size=512,  # Prior latent size.
    pretrained_type='with_stylegan2'):  # Pretrained type for G.

    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # Load training set.
    training_set = dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir), verbose=True, **dataset_args)
    grid_size, grid_reals, grid_labels = misc.setup_snapshot_image_grid(training_set, **grid_args)
    misc.save_image_grid(grid_reals, dnnlib.make_run_dir_path('reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)

    # Construct or load networks.
    with tf.device('/gpu:0'):
        if resume_pkl is None or resume_with_new_nets:
            print('Constructing networks...')
            I = tflib.Network('I', num_channels=training_set.shape[0], resolution=resolution_manual, label_size=training_set.label_size, **I_args)
            M = tflib.Network('M', num_channels=training_set.shape[0], resolution=resolution_manual, label_size=training_set.label_size, **M_args)
            Is = I.clone('Is')
            if use_hd_with_cls:
                I_info = tflib.Network('I_info', num_channels=training_set.shape[0], resolution=resolution_manual, label_size=training_set.label_size, **I_info_args)
        if resume_pkl is not None:
            print('Loading networks from "%s"...' % resume_pkl)
            if use_hd_with_cls:
                rI, rM, rIs, rI_info = misc.load_pkl(resume_pkl)
            else:
                rI, rM, rIs = misc.load_pkl(resume_pkl)
            if resume_with_new_nets:
                I.copy_vars_from(rI); M.copy_vars_from(rM); Is.copy_vars_from(rIs)
                if use_hd_with_cls:
                    I_info.copy_vars_from(rI_info)
            else:
                I = rI; M = rM; Is = rIs
                if use_hd_with_cls:
                    I_info =rI_info

        print('Loading generator from "%s"...' % resume_pkl)
        if pretrained_type == 'with_stylegan2':
            rG, rD, rGs = misc.load_pkl(resume_G_pkl)
            G = rG; D = rD; Gs = rGs
        elif pretrained_type == 'with_cascadeVAE':
            rG = misc.load_pkl(resume_G_pkl)
            G = rG; Gs = rG

    # Print layers and generate initial image snapshot.
    I.print_layers(); M.print_layers()
    # pdb.set_trace()
    training_set_resolution_log2 = int(np.log2(resolution_manual))
    sched = training_schedule(cur_nimg=total_kimg*1000, training_set_resolution_log2=training_set_resolution_log2, **sched_args)

    grid_size, grid_latents, grid_labels = get_grid_latents(
        n_discrete, n_continuous, n_samples_per, G, grid_labels)
    print('grid_size:', grid_size)
    print('grid_latents.shape:', grid_latents.shape)
    print('grid_labels.shape:', grid_labels.shape)
    if resolution_manual >= 256:
        grid_size = (grid_size[0], grid_size[1]//5)
        grid_latents = grid_latents[:grid_latents.shape[0]//5]
        grid_labels = grid_labels[:grid_labels.shape[0]//5]
    prior_traj_latents = M.run(grid_latents,
                        is_validation=True,
                        minibatch_size=sched.minibatch_gpu)
    if use_std_in_m:
        prior_traj_latents = prior_traj_latents[:, :prior_latent_size]
    prior_traj_latents_show = np.reshape(prior_traj_latents,
                                         [-1, n_samples_per, prior_latent_size])
    print_traj(prior_traj_latents_show)
    grid_fakes = Gs.run(prior_traj_latents,
                        grid_labels,
                        is_validation=True,
                        minibatch_size=sched.minibatch_gpu,
                        randomize_noise=True,
                        normalize_latents=False)
    misc.save_image_grid(grid_fakes,
                         dnnlib.make_run_dir_path('fakes_init.png'),
                         drange=drange_net,
                         grid_size=grid_size)

    if use_level_training:
        ending_level = training_set_resolution_log2 - 1
    else:
        ending_level = 1

    # Setup training inputs.
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in               = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in             = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_size_in    = tf.placeholder(tf.int32, name='minibatch_size_in', shape=[])
        minibatch_gpu_in     = tf.placeholder(tf.int32, name='minibatch_gpu_in', shape=[])
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
        Is_beta              = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), I_smoothing_kimg * 1000.0) if I_smoothing_kimg > 0.0 else 0.0

    # Setup optimizers.
    I_opt_args = dict(I_opt_args)
    for args, reg_interval in [(I_opt_args, I_reg_interval)]:
        args['minibatch_multiplier'] = minibatch_multiplier
        args['learning_rate'] = lrate_in
        if lazy_regularization:
            mb_ratio = reg_interval / (reg_interval + 1)
            args['learning_rate'] *= mb_ratio
            if 'beta1' in args: args['beta1'] **= mb_ratio
            if 'beta2' in args: args['beta2'] **= mb_ratio

    I_opts = []
    I_reg_opts = []
    for n_level in range(ending_level):
        I_opts.append(tflib.Optimizer(name='TrainI_%d' % n_level, **I_opt_args))
        I_reg_opts.append(tflib.Optimizer(name='RegI_%d' % n_level, share=I_opts[-1], **I_opt_args))

    # Build training graph for each GPU.
    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):

            # Create GPU-specific shadow copies of I and M.
            I_gpu = I if gpu == 0 else I.clone(I.name + '_shadow')
            M_gpu = M if gpu == 0 else M.clone(M.name + '_shadow')
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            if use_hd_with_cls:
                I_info_gpu = I_info if gpu == 0 else I_info.clone(I_info.name + '_shadow')

            # Evaluate loss functions.
            lod_assign_ops = []
            I_losses = []
            I_regs = []
            if 'lod' in I_gpu.vars: lod_assign_ops += [tf.assign(I_gpu.vars['lod'], lod_in)]
            if 'lod' in M_gpu.vars: lod_assign_ops += [tf.assign(M_gpu.vars['lod'], lod_in)]
            for n_level in range(ending_level):
                with tf.control_dependencies(lod_assign_ops):
                    with tf.name_scope('I_loss_%d' % n_level):
                        if use_hd_with_cls:
                            I_loss, I_reg = dnnlib.util.call_func_by_name(I=I_gpu, M=M_gpu, G=G_gpu, I_info=I_info_gpu, opt=I_opts[n_level], training_set=training_set, minibatch_size=minibatch_gpu_in, **I_loss_args)
                        else:
                            I_loss, I_reg = dnnlib.util.call_func_by_name(I=I_gpu, M=M_gpu, G=G_gpu, opt=I_opts[n_level], n_levels=(n_level + 1) if use_level_training else training_set_resolution_log2 - 1,
                                                                          training_set=training_set, minibatch_size=minibatch_gpu_in, **I_loss_args)
                        I_losses.append(I_loss)
                        I_regs.append(I_reg)

                # Register gradients.
                if not lazy_regularization:
                    if I_regs[n_level] is not None: I_losses[n_level] += I_regs[n_level]
                else:
                    if I_regs[n_level] is not None: I_reg_opts[n_level].register_gradients(tf.reduce_mean(I_regs[n_level] * I_reg_interval), I_gpu.trainables)

                if use_hd_with_cls:
                    MIIinfo_gpu_trainables = collections.OrderedDict(
                        list(M_gpu.trainables.items()) +
                        list(I_gpu.trainables.items()) +
                        list(I_info_gpu.trainables.items())
                    )
                    I_opts[n_level].register_gradients(tf.reduce_mean(I_losses[n_level]), MIIinfo_gpu_trainables)
                else:
                    MI_gpu_trainables = collections.OrderedDict(
                        list(M_gpu.trainables.items()) +
                        list(I_gpu.trainables.items()))
                    I_opts[n_level].register_gradients(tf.reduce_mean(I_losses[n_level]), MI_gpu_trainables)

    # Setup training ops.
    I_train_ops = []
    I_reg_ops = []
    for n_level in range(ending_level):
        I_train_ops.append(I_opts[n_level].apply_updates())
        I_reg_ops.append(I_reg_opts[n_level].apply_updates(allow_no_op=True))
    Is_update_op = Is.setup_as_moving_average_of(I, beta=Is_beta)

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
        I.setup_weight_histograms(); M.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    running_mb_counter = 0
    while cur_nimg < total_kimg * 1000:
        if dnnlib.RunContext.get().should_stop(): break

        n_level = 0 if not use_level_training else min(cur_nimg // (level_I_kimg * 1000), training_set_resolution_log2 - 2)
        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set_resolution_log2=training_set_resolution_log2, **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        training_set.configure(sched.minibatch_gpu, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                I_opts[n_level].reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        feed_dict = {lod_in: sched.lod, lrate_in: sched.I_lrate, minibatch_size_in: sched.minibatch_size, minibatch_gpu_in: sched.minibatch_gpu}
        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus)
            run_I_reg = (lazy_regularization and running_mb_counter % I_reg_interval == 0)
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1

            # Fast path without gradient accumulation.
            if len(rounds) == 1:
                tflib.run([I_train_ops[n_level]], feed_dict)
                if run_I_reg:
                    tflib.run(I_reg_ops[n_level], feed_dict)
                tflib.run([Is_update_op], feed_dict)

            # Slow path with gradient accumulation.
            else:
                for _round in rounds:
                    tflib.run(I_train_ops[n_level], feed_dict)
                if run_I_reg:
                    for _round in rounds:
                        tflib.run(I_reg_ops[n_level], feed_dict)
                tflib.run(Is_update_op, feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 100 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()
            total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', sched.lod),
                autosummary('Progress/minibatch', sched.minibatch_size),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if image_snapshot_ticks is not None and (cur_tick % image_snapshot_ticks == 0 or done):
                prior_traj_latents = M.run(grid_latents,
                                    is_validation=True,
                                    minibatch_size=sched.minibatch_gpu)
                if use_std_in_m:
                    prior_traj_latents = prior_traj_latents[:, :prior_latent_size]
                prior_traj_latents_show = np.reshape(prior_traj_latents, [-1, n_samples_per, prior_latent_size])
                print_traj(prior_traj_latents_show)
                grid_fakes = Gs.run(prior_traj_latents, grid_labels, is_validation=True,
                                    minibatch_size=sched.minibatch_gpu, randomize_noise=True, normalize_latents=False)
                misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
            if network_snapshot_ticks is not None and (cur_tick % network_snapshot_ticks == 0 or done):
                pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((I, M, Is), pkl)
                metrics.run(pkl, run_dir=dnnlib.make_run_dir_path(), data_dir=dnnlib.convert_path(data_dir), num_gpus=num_gpus, tf_config=tf_config)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            dnnlib.RunContext.get().update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time

    # Save final snapshot.
    misc.save_pkl((I, M, Is), dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()
    training_set.close()

#----------------------------------------------------------------------------
