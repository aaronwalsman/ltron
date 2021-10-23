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

    if os.path.exists(raw_annot_dir):
        with open(raw_annot_dir, 'r') as f:
            raw_annot = json.load(f)
    else:
        generate_json(raw_dir, "./")

    blacklist_out(raw_dir, whole_dest)
    print('blacklist done')
    # Partition the omr files into connected components
    partition_omr(whole_dest, omr_dest)
    print('partition done')
    # Replace the brick variants with standard bricks
    remap_variant_paths(omr_dest, omr_dest)
    print('remapping done')
    # Slice 8 bricks submodels
    subcomponent_nonoverlap_extraction(omr_dest, 8, float('inf'), blacklist=[], min_size=200, max_size=480)
    print('subcomponent extraction done')
    # Generate the annotation omr_clean.json
    generate_json(omr_dest, "./", mode='clean')
    generate_json(slice_dir, slice_dir, mode='clean')
    print('json generation done')
    # Generate sample rollouts of disassembly/reassembly
    rollout(omr_dest, rollout_dest)

if __name__ == '__main__':
    main()