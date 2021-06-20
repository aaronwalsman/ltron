#!/usr/bin/env python
import os
import argparse

import ltron.settings as settings
import ltron.license as license
import ltron.installation as installation

parser = argparse.ArgumentParser()

parser.add_argument('--resolution', type=str, default='low',
    help='resolution of the brick shapes, can be either "low" or "high"')
parser.add_argument('--overwrite', action='store_true',
    help='if this flag is specified, any existing data will be removed')

def main():
    args = parser.parse_args()

    installation.make_settings_cfg(overwrite=args.overwrite)
    settings.reload_settings()
    installation.install_ldraw(overwrite=args.overwrite)
    installation.install_splendor_meshes(args.resolution)
    installation.install_ldcad(overwrite=args.overwrite)
    installation.install_collection('omr', overwrite=args.overwrite)
    installation.install_collection('random_six', overwrite=args.overwrite)
