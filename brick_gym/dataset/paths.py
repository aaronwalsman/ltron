import math
import os
import glob
import json

import brick_gym.config as config

def resolve_subdocument(file_path):
    if ':' in file_path:
        file_path, subdocument = file_path.split(':')
    else:
        subdocument = None
    
    return file_path, subdocument

def get_metadata_path(file_path):
    file_path = os.path.expanduser(file_path)
    directory, file_name = os.path.split(file_path)
    file_basename = os.path.splitext(file_name)[0]
    primary_index = file_basename.split('_')[1]
    metadata_path = os.path.join(directory, 'metadata_%s.json'%primary_index)
    
    return metadata_path

def get_metadata(file_path):
    metadata_path = get_metadata_path(file_path)
    metadata = json.load(open(metadata_path))
    return metadata

def get_dataset_info(dataset):
    #dataset_directory = os.path.expanduser(config.datasets[dataset])
    #dataset_path = os.path.join(dataset_directory, 'dataset.json')
    dataset_path = os.path.expanduser(config.datasets[dataset])
    return json.load(open(dataset_path))

def get_dataset_paths(dataset, split_name, subset=None, rank=0, size=1):
    dataset_path = os.path.expanduser(config.datasets[dataset])
    dataset_directory = os.path.dirname(dataset_path)
    splits = get_dataset_info(dataset)['splits']
    split_globs = splits[split_name]
    all_file_paths = []
    for split_glob in split_globs:
        if ':' in split_glob:
            split_glob, sub_model = split_glob
        else:
            sub_model = None
        
        file_paths = glob.glob(os.path.join(
                dataset_directory, split_glob))
        if sub_model is not None:
            file_paths = ['%s:%s'%(fp, sub_model) for fp in file_paths]
        all_file_paths.extend(file_paths)
    all_file_paths.sort()
    if subset is not None:
        if isinstance(subset, int):
            subset = (subset,)
        all_file_paths = all_file_paths[slice(*subset)]
    
    #stride = math.ceil(len(all_file_paths) / size)
    #paths = all_file_paths[rank*stride:(rank+1)*stride]
    paths = all_file_paths[rank::size]
    return paths
