import math
import os
from zipfile import ZipFile
import glob
import json

import numpy

import ltron.settings as settings
from ltron.hierarchy import map_hierarchies, concatenate_lists

def resolve_subdocument(file_path):
    if '#' in file_path:
        file_path, subdocument = file_path.split('#')
        subdocument = subdocument.lower()
    else:
        subdocument = None
    
    file_path = os.path.expanduser(file_path)
    
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
    #dataset_directory = os.path.expanduser(settings.datasets[dataset])
    #dataset_path = os.path.join(dataset_directory, 'dataset.json')
    dataset_path = os.path.expanduser(settings.datasets[dataset])
    return json.load(open(dataset_path))

def get_subset_slice(subset):
    if subset is None:
        return slice(None)
    
    if isinstance(subset, int):
        s = (subset,)
    else:
        s = subset
    return slice(*s)

def process_file_paths(file_paths, subset=None, rank=0, size=1):
    file_paths = file_paths.split(',')
    all_file_paths = []
    for file_path in file_paths:
        file_path = file_path.format(**settings.collections)
        file_path, subdocument = resolve_subdocument(file_path)
        if '[' in file_path:
            file_path, path_slice = file_path.split('[')
            path_slice = path_slice.replace(']', '').split(':')
            path_slice = [None if s == '' else int(s) for s in path_slice]
            path_slice = slice(*path_slice)
        else:
            path_slice = slice(None)
        
        file_paths = sorted(glob.glob(file_path))[path_slice]
        if subdocument is not None:
            file_paths = ['%s#%s'%(fp, subdocument) for fp in file_paths]
        all_file_paths.extend(file_paths)
    
    if subset is not None:
        #if isinstance(subset, int):
        #    path_subset = (subset,)
        #else:
        #    path_subset = subset
        s = get_subset_slice(subset)
        all_file_paths = all_file_paths[s]
    
    paths = all_file_paths[rank::size]
    return numpy.array(paths, dtype=object)

def get_dataset_paths(dataset, split_name, subset=None, rank=0, size=1):
    split = get_dataset_info(dataset)['splits'][split_name]
    
    def process_fn(file_paths):
        return process_file_paths(
            file_paths, subset=subset, rank=rank, size=size)
    
    paths = map_hierarchies(process_fn, split)
    return concatenate_lists(paths)

def get_zip_paths(dataset, split_name, subset=None, key='zip', rank=0, size=1):
    zip_path = get_dataset_info(dataset)['splits'][split_name][key]
    zip_path = zip_path.format(**settings.collections)
    z = ZipFile(zip_path, 'r')
    names = [info.filename for info in z.infolist() if not info.is_dir()]
    names = names[get_subset_slice(subset)]
    return z, names

def get_dataset_paths_old(dataset, split_name, subset=None, rank=0, size=1):
    dataset_path = os.path.expanduser(settings.datasets[dataset])
    dataset_directory = os.path.dirname(dataset_path)
    splits = get_dataset_info(dataset)['splits']
    split_globs = splits[split_name]
    all_file_paths = []
    for split_glob in split_globs:
        if ':' in split_glob:
            split_glob, sub_model = split_glob.split(':')
        else:
            sub_model = None
        
        split_glob = split_glob.format(**settings.collections)
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
    
    paths = all_file_paths[rank::size]
    return paths
