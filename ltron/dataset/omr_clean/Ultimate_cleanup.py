from brick_variants import remap_variant_paths
from ltron.dataset.datastructure.connected_components import partition_omr
from rollouts_clean import rollout
from dataset_annotation import generate_json
from blacklist import blacklist_out
from ltron.dataset.submodel_extraction import subcomponent_nonoverlap_extraction
import sys
import os
import json

# Overall pipeline
def main():

    raw_dir = "~/.cache/ltron/collections/omr/ldraw/"
    raw_annot_dir = "omr_raw.json"
    omr_dest = "ldraw/"
    whole_dest = "whole/"
    rollout_dest = "rollouts/"
    slice_dir = "subcomponents8"
    slice_dir2 = "subcomponents2"
    slice_dir32 = "subcomponents32"
    slice_dir128 = "subcomponents128"

    if os.path.exists(raw_annot_dir):
        with open(raw_annot_dir, 'r') as f:
            raw_annot = json.load(f)
    else:
        generate_json(raw_dir, "./")

    with open("blacklist_70.json", 'r') as f:
        bl = json.load(f)
    # # subcomponent_nonoverlap_extraction(raw_dir, 8, float('inf'), blacklist=bl, min_size=200, max_size=480, folder_name="raw_subcomponents8/")
    # pre_blacklist = ['3626.dat', '3625.dat', '3624.dat', '41879.dat', '3820.dat', '3819.dat', '3818.dat', '10048.dat', '973.dat', '3817.dat', '3816.dat', '3815.dat', '92198.dat',
    #                  '92250.dat', '2599.dat', '93352.dat', '92245.dat', '92244.dat', '92248.dat', '92257.dat', '92251.dat', '92256.dat', '92241.dat', '92252.dat', '92258.dat', '92255.dat',
    #                  '93352.dat', '92259.dat', '92243.dat', '62810.dat', '92240.dat', '92438.dat', '92242.dat', '92247.dat', '92251.dat', '87991.dat', 'u9201', '95227.dat']
    # blacklist_out(raw_dir, whole_dest, threshold=400, blacklist = pre_blacklist)
    # print('blacklist done')
    # subcomponent_nonoverlap_extraction(whole_dest, 8, float('inf'), blacklist=[], min_size=200, max_size=480,
    #                                    folder_name="whole_subcomponents8/")
    # # Partition the omr files into connected components
    # partition_omr(whole_dest, omr_dest, remove_thre=1)
    # print('partition done')
    # # Replace the brick variants with standard bricks
    # remap_variant_paths(omr_dest, omr_dest, overwrite=True)
    # print('remapping done')
    # # Slice 8 bricks submodels
    # subcomponent_nonoverlap_extraction(omr_dest, 8, float('inf'), blacklist=[], min_size=200, max_size=400)
    # subcomponent_nonoverlap_extraction(omr_dest, 2, float('inf'), blacklist=[], min_size=200, max_size=400)
    # subcomponent_nonoverlap_extraction(omr_dest, 32, float('inf'), blacklist=[], min_size=200, max_size=400)
    # subcomponent_nonoverlap_extraction(omr_dest, 128, float('inf'), blacklist=[], min_size=200, max_size=400)
    # print('subcomponent extraction done')
    # # Generate the annotation omr_clean.json
    generate_json(omr_dest, "./", mode='clean')
    generate_json(slice_dir2, "./", mode='clean')
    generate_json(slice_dir32, "./", mode='clean')
    generate_json(slice_dir128, "./", mode='clean')
    print('json generation done')
    # Generate sample rollouts of disassembly/reassembly
    rollout(omr_dest, rollout_dest)

if __name__ == '__main__':
    main()