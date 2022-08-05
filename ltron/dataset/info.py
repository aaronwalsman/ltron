import os
import json

import ltron.settings as settings
from ltron.exceptions import LtronMissingDatasetException

#def get_dataset_path(dataset):
#    return os.path.expanduser(
#        os.path.join(settings.paths['dataset'], '%s.json'%dataset))

#def get_shard_path(shard):
#    return os.path.expanduser(os.path.join(settings.paths['shards'], shard))

def get_dataset_info(dataset):
    try:
        #return json.load(open(get_dataset_path(dataset_path)))
        return json.load(open(settings.datasets[dataset]))
    except KeyError:
        raise LtronMissingDatasetException(dataset)

def get_split_shards(dataset, split):
    info = get_dataset_info(dataset)
    return info['splits'][split]['shards']

#def build_dataset_info(
#    dataset,
#    brick_shapes,
#    brick_colors,
#    max_instances_per_scene,
#    max_edges_per_scene,
#    splits,
#):
#    info = {}
#    
#    brick_colors = sorted(brick_colors)
#    info['color_ids'] = dict(zip(brick_colors, range(1, len(brick_colors)+1)))
#    
#    brick_shapes = list(sorted(s for s in brick_shapes))
#    info['shape_ids'] = dict(zip(brick_shapes, range(1, len(brick_shapes)+1)))
#    
#    info['max_instances_per_scene'] = max_instances_per_scene
#    info['max_edges_per_scene'] = max_edges_per_scene
#    
#    info['splits'] = splits
#    
#    dataset_path = os.path.expanduser(get_dataset_path(dataset))
#    with open(dataset_path, 'w') as f:
#        json.dump(info, f, indent=2)
