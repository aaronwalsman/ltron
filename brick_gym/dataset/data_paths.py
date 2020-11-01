import math
import os
import glob
import json

def data_paths(self,
        dataset_directory,
        split_name,
        subset=None,
        rank=0,
        size=1):
    dataset_directory = os.path.expanduser(dataset_directory)
    split_file = os.path.join(dataset_directory, 'splits.json')
    with open(split_file, 'r') as f:
        splits = json.load(f)
    split_glob = splits[split_name]
    
    all_file_paths = sorted(glob.glob(os.path.join(
            dataset_directory, split_glob)))
    if subset is not None:
        if isinstance(subset, int):
            subset = (subset,)
        all_file_paths = all_file_paths[slice(*subset)]
    
    stride = math.ceil(len(all_file_paths) / size)
    return all_file_paths[rank*stride, (rank+1)*stride]
