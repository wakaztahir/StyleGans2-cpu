#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: convert_vc2.py
# --- Creation Date: 19-03-2021
# --- Last Modified: Fri 19 Mar 2021 15:00:16 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Convert a pkl to a new model.
"""
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os
import collections
import cv2

import pretrained_networks
from training import misc
from training.utils import get_grid_latents, get_return_v, add_outline, save_atts
from run_editing_vc2 import image_to_ready
from run_editing_vc2 import image_to_out
from PIL import Image, ImageDraw, ImageFont

#----------------------------------------------------------------------------

def convert_pkl(network_pkl, new_func_name_G, new_func_name_D, new_func_name_I):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    _G, _D, _I, _Gs = misc.load_pkl(network_pkl)
    Gs = _Gs.convert(new_func_name=new_func_name_G, synthesis_func='G_synthesis_modular_ps_sc')
    G = _G.convert(new_func_name=new_func_name_G, synthesis_func='G_synthesis_modular_ps_sc')
    D = _D.convert(new_func_name=new_func_name_D)
    I = _I.convert(new_func_name=new_func_name_I)

    misc.save_pkl((G, D, I, Gs),
                  dnnlib.make_run_dir_path('network-saved.pkl'))


def main():
    parser = argparse.ArgumentParser(description='''Convert pkl.''')

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('convert', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_generate_images.add_argument('--new_func_name_G', help='new G func name if create new G', default='training.ps_sc_networks2.G_main_ps_sc')
    parser_generate_images.add_argument('--new_func_name_D', help='new D func name if create new D', default='training.networks_stylegan2.D_stylegan2')
    parser_generate_images.add_argument('--new_func_name_I', help='new I func name if create new I', default='training.ps_sc_networks2.head_ps_sc')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'convert': 'convert_vc2.convert_pkl',
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
