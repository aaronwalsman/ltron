#!/usr/bin/env python
import os
import argparse

from splendor.assets import install_assets
from splendor.scripts import splendor_asset_installer

import ltron.settings as settings
import ltron.license as license
import ltron.installation as installation
import ltron.home as home

parser = argparse.ArgumentParser()

parser.add_argument('--resolution', type=str, default='low',
    help='resolution of the brick shapes, can be either "low" or "high"')
parser.add_argument('--overwrite', action='store_true',
    help='if this flag is specified, any existing data will be removed')
parser.add_argument('--install-episodes', action='store_true',
    help='install expert-generated episodes in addition to the ldraw files')
parser.add_argument('--install-lstm-pretrained-weights', action='store_true',
    help='install pretrained weights for the ECCV lstm experiments')

def main():
    args = parser.parse_args()
    
    home.make_ltron_home()
    
    install_assets(
        splendor_asset_installer.asset_urls['default_assets'], 'default_assets')
    installation.make_settings_cfg(overwrite=args.overwrite)
    #installation.make_blacklist_json(overwrite=args.overwrite)
    settings.reload_settings()
    installation.install_ldraw(overwrite=args.overwrite)
    installation.install_splendor_meshes(args.resolution)
    installation.install_ldcad(overwrite=args.overwrite)
    installation.install_collection(
        'omr', overwrite=args.overwrite)
    installation.install_collection(
        'omr_clean', overwrite=args.overwrite)
    installation.install_collection(
        'random_construction_6_6', overwrite=args.overwrite)
    installation.install_extras(overwrite=args.overwrite)
    if args.install_episodes:
        installation.install_episodes(
            'omr_clean',
            'omr_episodes_2',
            overwrite=args.overwrite,
        )
        installation.install_episodes(
            'omr_clean',
            'omr_episodes_4',
            overwrite=args.overwrite,
        )
        installation.install_episodes(
            'omr_clean',
            'omr_episodes_8',
            overwrite=args.overwrite,
        )
        installation.install_episodes(
            'random_construction_6_6',
            'random_construction_6_6_episodes_2',
            overwrite=args.overwrite,
        )
        installation.install_episodes(
            'random_construction_6_6',
            'random_construction_6_6_episodes_4',
            overwrite=args.overwrite,
        )
        installation.install_episodes(
            'random_construction_6_6',
            'random_construction_6_6_episodes_8',
            overwrite=args.overwrite,
        )
    if args.install_lstm_pretrained_weights:
        installation.install_pretrained_lstm_weights(overwrite=args.overwrite)
