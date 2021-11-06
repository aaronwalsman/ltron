from ltron.dataset.omr_clean.brick_variants import remap_variant_paths
from ltron.dataset.datastructure.connected_components import partition_omr
#from rollouts_clean import rollout
from ltron.dataset.omr_clean.dataset_annotation import generate_json
from ltron.dataset.omr_clean.blacklist import blacklist_out
from ltron.dataset.submodel_extraction import subcomponent_nonoverlap_extraction
from ltron.home import get_ltron_home
import ltron.settings as settings
import sys
import os
import json
from glob import glob
import shutil

# Overall pipeline
def main():

    print('='*80)
    print('Setup')
    home = get_ltron_home()
    #raw_dir = "~/.cache/ltron/collections/omr/ldraw/"
    raw_dir = os.path.join(settings.collections['omr'], 'ldraw')
    raw_annot_dir = "omr_raw.json"
    omr_clean = settings.collections['omr_clean']
    if not os.path.exists(omr_clean):
        os.makedirs(omr_clean)
    omr_dest = os.path.join(omr_clean, "ldraw/")
    if not os.path.exists(omr_dest):
        os.makedirs(omr_dest)
    whole_dest = os.path.join(omr_clean, "whole/")
    if not os.path.exists(whole_dest):
        os.makedirs(whole_dest)
    rollout_dest = os.path.join(omr_clean, "rollouts/")
    if not os.path.exists(rollout_dest):
        os.makedirs(rollout_dest)
    #slice_dir = os.path.join(omr_clean, "subcomponents8")
    #slice_dir2 = os.path.join(omr_clean, "subcomponents2")
    #slice_dir32 = os.path.join(omr_clean, "subcomponents32")
    #slice_dir128 = os.path.join(omr_clean, "subcomponents128")
    slice_dir2 = omr_dest
    slice_dir8 = omr_dest
    slice_dir32 = omr_dest
    slice_dir128 = omr_dest
    
    if os.path.exists(raw_annot_dir):
        with open(raw_annot_dir, 'r') as f:
            raw_annot = json.load(f)
    else:
        generate_json(raw_dir, "./")
    
    # remove all blacklisted bricks from the raw files and save them in whole
    pre_blacklist = ['3626.dat', '3625.dat', '3624.dat', '41879.dat',
        '3820.dat', '3819.dat', '3818.dat', '10048.dat', '973.dat',
        '3817.dat', '3816.dat', '3815.dat', '92198.dat', '92250.dat',
        '2599.dat', '93352.dat', '92245.dat', '92244.dat', '92248.dat',
        '92257.dat', '92251.dat', '92256.dat', '92241.dat', '92252.dat',
        '92258.dat', '92255.dat', '93352.dat', '92259.dat', '92243.dat',
        '62810.dat', '92240.dat', '92438.dat', '92242.dat', '92247.dat',
        '92251.dat', '87991.dat', 'u9201.dat', '95227.dat'
    ]
    
    print('='*80)
    print('Brick Blacklist')
    blacklist_out(
        raw_dir, whole_dest, omr_clean, threshold=400, blacklist=pre_blacklist)
    
    # Replace the brick variants with standard bricks
    print('='*80)
    print('Remapping Variants')
    remap_variant_paths(whole_dest, whole_dest, overwrite=True)
    
    # Partition the omr files into connected components
    print('='*80)
    print('Partitioning Into Connected Components')
    partition_omr(whole_dest, omr_dest, remove_thre=1)
    
    # Slice and dice
    print('='*80)
    print('Slicing')
    print('-'*80)
    print('Building 2 Brick Slices')
    folder2 = os.path.join(omr_clean, 'slice2')
    subcomponent_nonoverlap_extraction(
        omr_dest, 2, float('inf'), blacklist=[], min_size=200, max_size=400,
        folder_name=folder2)
    print('-'*80)
    print('Building 8 Brick Slices')
    folder8 = os.path.join(omr_clean, 'slice8')
    subcomponent_nonoverlap_extraction(
        omr_dest, 8, float('inf'), blacklist=[], min_size=200, max_size=400,
        folder_name=folder8)
    print('-'*80)
    print('Building 32 Brick Slices')
    folder32 = os.path.join(omr_clean, 'slice32')
    subcomponent_nonoverlap_extraction(
        omr_dest, 32, float('inf'), blacklist=[], min_size=200, max_size=400,
        folder_name=folder32)
    print('-'*80)
    print('Building 128 Brick Slices')
    folder128 = os.path.join(omr_clean, 'slice128')
    subcomponent_nonoverlap_extraction(
        omr_dest, 128, float('inf'), blacklist=[], min_size=200, max_size=400,
        folder_name=folder128)
    
    for f in glob(os.path.join(folder2, '*')):
        shutil.move(f, omr_dest)
    shutil.rmtree(folder2)
    for f in glob(os.path.join(folder8, '*')):
        shutil.move(f, omr_dest)
    shutil.rmtree(folder8)
    for f in glob(os.path.join(folder32, '*')):
        shutil.move(f, omr_dest)
    shutil.rmtree(folder32)
    for f in glob(os.path.join(folder128, '*')):
        shutil.move(f, omr_dest)
    shutil.rmtree(folder128)
    
    
    # Generate the annotation omr_clean.json
    print('='*80)
    print('Generating json')
    # PUTS JSON FILE IN CURRENT DIRECTORY RATHER THAN DESTINATION DIRECTORY
    generate_json(omr_dest, "./", mode='clean')
    #generate_json(slice_dir2, "./", mode='clean')
    #generate_json(slice_dir8, "./", mode='clean')
    #generate_json(slice_dir32, "./", mode='clean')
    #generate_json(slice_dir128, "./", mode='clean')
    # Generate sample rollouts of disassembly/reassembly
    #rollout(omr_dest, rollout_dest)

if __name__ == '__main__':
    main()
