import argparse

import ltron.geometry.symmetry as symmetry
from ltron.home import get_ltron_home

parser = argparse.ArgumentParser()

parser.add_argument(
    '--symmetry-table-path', type=str,
    default=symmetry.symmetry_table_path)
parser.add_argument(
    '--resolution', type=int, default=symmetry.default_resolution)
parser.add_argument(
    '--tolerance', type=int, default=symmetry.default_tolerance)
parser.add_argument(
    '--bricks', type=str, default=None)
parser.add_argument(
    '--error-handling', type=str, default='skip')

def main():
    args = parser.parse_args()
    
    bricks = args.bricks
    if bricks is not None:
        bricks = bricks.split(',')
    
    symmetry.build_symmetry_table(
        bricks=bricks,
        symmetry_table_path=args.symmetry_table_path,
        resolution=args.resolution,
        tolerance=args.tolerance,
        error_handling=args.error_handling,
    )
