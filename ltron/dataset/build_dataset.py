import random
import json
import os

import tqdm

from ltron.bricks.brick_scene import BrickScene

random.seed(141414)

def build_dataset(name, path_root, paths, test_set):
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
        brick_names = set(scene.brick_library.keys())
        all_brick_names |= brick_names
        
        num_instances = len(scene.instances)
        max_instances_per_scene = max(max_instances_per_scene, num_instances)
        
        edges = scene.get_all_edges(unidirectional=True)
        num_edges = edges.shape[1]
        max_edges_per_scene = max(max_edges_per_scene, num_edges)
        colors = set(scene.color_library.keys())
        all_colors |= colors
        
        for instance_id, instance in scene.instances.items():
            brick_name = str(instance.brick_type)
            if brick_name not in instance_counts:
                instance_counts[brick_name] = 0
            instance_counts[brick_name] += 1
    
    data['max_instances_per_scene'] = max_instances_per_scene
    data['max_edges_per_scene'] = max_edges_per_scene
    data['class_ids'] = dict(zip(
            sorted(instance_counts.keys()),
            range(1, len(instance_counts)+1)))
    data['all_colors'] = list(sorted(all_colors, key=int))
    
    with open(os.path.join(path_root, f'{name}.json'), 'w') as output_file:
        json.dump(data, output_file, indent=2)
