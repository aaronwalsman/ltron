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

def main():
    args = parser.parse_args()
    
    home.make_ltron_home()
    
    install_assets(
        splendor_asset_installer.asset_urls['default_assets'], 'default_assets')
    installation.make_settings_cfg(overwrite=args.overwrite)
    installation.make_blacklist_json(overwrite=args.overwrite)
    settings.reload_settings()
    installation.install_ldraw(overwrite=args.overwrite)
    installation.install_splendor_meshes(args.resolution)
    installation.install_ldcad(overwrite=args.overwrite)
    installation.install_collection('omr', overwrite=args.overwrite)
    installation.install_collection('random_six', overwrite=args.overwrite)
