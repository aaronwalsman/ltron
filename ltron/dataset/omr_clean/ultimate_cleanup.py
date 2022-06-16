from argparse import ArgumentParser

from ltron.dataset.omr_clean.brick_variants import remap_variant_paths
from ltron.dataset.datastructure.connected_components import partition_omr
#from rollouts_clean import rollout
from ltron.dataset.omr_clean.dataset_annotation import generate_json
from ltron.dataset.omr_clean.blacklist import remove_blacklisted_parts
from ltron.dataset.submodel_extraction import subcomponent_nonoverlap_extraction
from ltron.home import get_ltron_home
import ltron.settings as settings
import sys
import os
import json
from glob import glob
import shutil

# Overall pipeline
def clean_omr():
    
    # get arguments
    parser = ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    overwrite = args.overwrite

    print('='*80)
    print('Setup')
    home = get_ltron_home()
    #raw_dir = "~/.cache/ltron/collections/omr/ldraw/"
    raw_dir = os.path.join(settings.collections['omr'], 'ldraw')
    raw_dataset_info_path = os.path.join(
        settings.collections['omr'], "omr.json")
    omr_clean = settings.collections['omr_clean']
    if not os.path.exists(omr_clean):
        os.makedirs(omr_clean)
    ldraw_dest = os.path.join(omr_clean, "ldraw/")
    if not os.path.exists(ldraw_dest):
        os.makedirs(ldraw_dest)
    slice_dests = {}
    for i in [2,4,8,32]:
        slice_dest = os.path.join(omr_clean, 'ldraw_%i'%i)
        if not os.path.exists(slice_dest):
            os.makedirs(slice_dest)
        slice_dests[i] = slice_dest
    remapped_dest = os.path.join(omr_clean, 'remapped/')
    if not os.path.exists(remapped_dest):
        os.makedirs(remapped_dest)
    whole_dest = os.path.join(omr_clean, "whole/")
    if not os.path.exists(whole_dest):
        os.makedirs(whole_dest)
    component_dest = os.path.join(omr_clean, 'components/')
    if not os.path.exists(component_dest):
        os.makedirs(component_dest)
    episode_dest = os.path.join(omr_clean, "episodes/")
    if not os.path.exists(episode_dest):
        os.makedirs(episode_dest)
    #slice_dir = os.path.join(omr_clean, "subcomponents8")
    #slice_dir2 = os.path.join(omr_clean, "subcomponents2")
    #slice_dir32 = os.path.join(omr_clean, "subcomponents32")
    #slice_dir128 = os.path.join(omr_clean, "subcomponents128")
    #slice_dir2 = ldraw_dest
    #slice_dir4 = ldraw_dest
    #slice_dir8 = ldraw_dest
    #slice_dir32 = ldraw_dest
    #slice_dir128 = ldraw_dest
    
    if os.path.exists(raw_dataset_info_path):
        with open(raw_dataset_info_path, 'r') as f:
            raw_annot = json.load(f)
    else:
        generate_json(raw_dir, "./")
    
    # remove all blacklisted bricks from the raw files and save them in whole
    # moved to blacklist.json
    #pre_blacklist = ['3626.dat', '3625.dat', '3624.dat', '41879.dat',
    #    '3820.dat', '3819.dat', '3818.dat', '10048.dat', '973.dat',
    #    '3817.dat', '3816.dat', '3815.dat', '92198.dat', '92250.dat',
    #    '2599.dat', '93352.dat', '92245.dat', '92244.dat', '92248.dat',
    #    '92257.dat', '92251.dat', '92256.dat', '92241.dat', '92252.dat',
    #    '92258.dat', '92255.dat', '93352.dat', '92259.dat', '92243.dat',
    #    '62810.dat', '92240.dat', '92438.dat', '92242.dat', '92247.dat',
    #    '92251.dat', '87991.dat', 'u9201.dat', '95227.dat'
    #]
    #pre_blacklist = []
    
    # Replace the brick variants with standard bricks
    print('='*80)
    print('Remapping Variants')
    remap_variant_paths(raw_dir, remapped_dest, overwrite=overwrite)
    
    print('='*80)
    print('Brick Blacklist')
    remove_blacklisted_parts(
        remapped_dest, whole_dest, threshold=400, overwrite=overwrite)
    
    # AARON TURNING THIS OFF FOR A MOMENT
    '''
    # Partition the omr files into connected components
    print('='*80)
    print('Partitioning Into Connected Components')
    partition_omr(whole_dest, component_dest, remove_threshold=1)
    '''
    
    folder2 = os.path.join(omr_clean, 'slice2')
    folder4 = os.path.join(omr_clean, 'slice4')
    folder8 = os.path.join(omr_clean, 'slice8')
    folder32 = os.path.join(omr_clean, 'slice32')
    # AARON TURNING THIS OFF FOR A MOMENT
    '''
    # Slice and dice
    print('='*80)
    print('Slicing')
    print('-'*80)
    print('Building 2 Brick Slices')
    folder2 = os.path.join(omr_clean, 'slice2')
    subcomponent_nonoverlap_extraction(
        component_dest,
        2, float('inf'), blacklist=[], min_size=200, max_size=400,
        folder_name=folder2)
    
    print('-'*80)
    print('Building 4 Brick Slices')
    folder8 = os.path.join(omr_clean, 'slice4')
    subcomponent_nonoverlap_extraction(
        component_dest,
        4, float('inf'), blacklist=[], min_size=200, max_size=400,
        folder_name=folder8)
    print('-'*80)
    
    print('-'*80)
    print('Building 8 Brick Slices')
    folder8 = os.path.join(omr_clean, 'slice8')
    subcomponent_nonoverlap_extraction(
        component_dest,
        8, float('inf'), blacklist=[], min_size=200, max_size=400,
        folder_name=folder8)
    print('-'*80)
    print('Building 32 Brick Slices')
    
    folder32 = os.path.join(omr_clean, 'slice32')
    subcomponent_nonoverlap_extraction(
        component_dest,
        32, float('inf'), blacklist=[], min_size=200, max_size=400,
        folder_name=folder32)
    print('-'*80)
    print('Building 128 Brick Slices')
    folder128 = os.path.join(omr_clean, 'slice128')
    subcomponent_nonoverlap_extraction(
        component_dest,
        128, float('inf'), blacklist=[], min_size=200, max_size=400,
        folder_name=folder128)
    '''
    
    '''
    for f in glob(os.path.join(folder2, '*')):
        shutil.move(f, slice_dir2)
    shutil.rmtree(folder2)
    for f in glob(os.path.join(folder8, '*')):
        shutil.move(f, slice_dir8)
    shutil.rmtree(folder8)
    for f in glob(os.path.join(folder32, '*')):
        shutil.move(f, slice_dir32)
    shutil.rmtree(folder32)
    for f in glob(os.path.join(folder128, '*')):
        shutil.move(f, slice_dir128)
    shutil.rmtree(folder128)
    '''
    
    # Generate the annotation omr_clean.json
    print('='*80)
    print('Generating json')
    # PUTS JSON FILE IN CURRENT DIRECTORY RATHER THAN DESTINATION DIRECTORY
    generate_json(ldraw_dest, "./", mode='clean')
    #generate_json(slice_dir2, "./", mode='clean')
    #generate_json(slice_dir8, "./", mode='clean')
    #generate_json(slice_dir32, "./", mode='clean')
    #generate_json(slice_dir128, "./", mode='clean')
    # Generate sample rollouts of disassembly/reassembly
    #rollout(omr_dest, rollout_dest)
