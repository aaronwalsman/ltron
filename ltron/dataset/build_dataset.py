import random
import json
import os
import glob

import tqdm

from ltron.bricks.brick_scene import BrickScene

random.seed(141414)

def build_metadata(name, path_root, test_percent):
    metadata = {}
    mpds = glob.glob(os.path.join(path_root, 'ldraw', '*.mpd'))
    num_test = round(len(mpds) * test_percent)
    num_train = len(mpds) - num_test
    metadata['splits'] = {
        'all':'{%s}/ldraw/*.mpd'%name,
        'train':'{%s}/ldraw/*.mpd[:%i]'%(name, num_train),
        'test':'{%s}/ldraw/*.mpd[%i:]'%(name, num_train),
    }
    
    max_instances_per_scene = 0
    max_edges_per_scene = 0
    all_brick_names = set()
    all_color_names = set()
    for mpd in tqdm.tqdm(mpds):
        scene = BrickScene(track_snaps=True)
        scene.import_ldraw(mpd)
        brick_names = set(scene.shape_library.keys())
        all_brick_names |= brick_names
        color_names = set(scene.color_library.keys())
        all_color_names |= color_names
        
        num_instances = len(scene.instances)
        max_instances_per_scene = max(num_instances, max_instances_per_scene)
        
        edges = scene.get_assembly_edges(unidirectional=False)
        num_edges = edges.shape[1]
        max_edges_per_scene = max(num_edges, max_edges_per_scene)
    
    metadata['max_instances_per_scene'] = max_instances_per_scene
    metadata['max_edges_per_scene'] = max_edges_per_scene
    metadata['shape_ids'] = {
        brick_name : i
        for i, brick_name in enumerate(all_brick_names, start=1)
    }
    metadata['color_ids'] = {
        color_name : i
        for i, color_name in enumerate(all_color_names, start=0)
    }
    
    out_path = os.path.join(path_root, '%s.json'%name)
    with open(out_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def build_dataset_old(name, path_root, paths, test_set):
    # intialize data
    data = {}
    data['splits'] = {}
    
    # generate splits
    relative_paths = [os.path.join('ldraw', path) for path in paths]
    
    data['splits']['all'] = list(sorted(relative_paths))
    
    if isinstance(test_set, int):
        test_paths = random.sample(relative_paths, test_set)
    else:
        test_paths = test_set
    data['splits']['test'] = list(sorted(test_paths))
    
    train_paths = set(relative_paths) - set(test_paths)
    data['splits']['train'] = list(sorted(train_paths))
    
    absolute_paths = [os.path.join(path_root, path) for path in relative_paths]
    
    # get class ids
    all_brick_names = set()
    all_colors = set()
    max_instances_per_scene = 0
    max_edges_per_scene = 0
    instance_counts = {}
    for path in tqdm.tqdm(absolute_paths):
        scene = BrickScene(track_snaps=True)
        scene.import_ldraw(path)
        brick_names = set(scene.shape_library.keys())
        all_brick_names |= brick_names
        
        num_instances = len(scene.instances)
        max_instances_per_scene = max(max_instances_per_scene, num_instances)
        
        edges = scene.get_assembly_edges(unidirectional=True)
        num_edges = edges.shape[1]
        max_edges_per_scene = max(max_edges_per_scene, num_edges)
        colors = set(scene.color_library.keys())
        all_colors |= colors
        
        for instance_id, instance in scene.instances.items():
            brick_name = str(instance.brick_shape)
            if brick_name not in instance_counts:
                instance_counts[brick_name] = 0
            instance_counts[brick_name] += 1
    
    data['max_instances_per_scene'] = max_instances_per_scene
    data['max_edges_per_scene'] = max_edges_per_scene
    data['shape_ids'] = dict(zip(
            sorted(instance_counts.keys()),
            range(1, len(instance_counts)+1)))
    data['all_colors'] = list(sorted(all_colors, key=int))
    
    with open(os.path.join(path_root, f'{name}.json'), 'w') as output_file:
        json.dump(data, output_file, indent=2)
