from argparse import ArgumentParser

import ltron.dataset.rca as rca

modules = {
    'rca':rca,
}
known_datasets = modules.keys()

parser = ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--size', type=str, nargs='*', default='all')
parser.add_argument('--split', type=str, nargs='*', default='all')
#parser.add_argument('--info', action='store_true')

def build_rc_dataset():
    args = parser.parse_args()
    
    module = modules[args.dataset]
    
    if args.size == 'all':
        sizes = module.all_sizes
    else:
        sizes = args.size
    
    if args.split == 'all':
        splits = module.all_splits
    else:
        splits = args.split
    
    collections = []
    for size in sizes:
        min_instances, max_instances = size.split('_')
        min_instances = int(min_instances)
        max_instances = int(max_instances)
        for split in splits:
            collections.append((min_instances, max_instances, split))
    
    module.build(collections)
    
    #if args.info:
    #    #splits = {}
    #    #all_sizes = module.all_sizes
    #    #all_splits = module.all_splits
    #    #splits['all'] = {
    #    #    'shards' : [
    #    #        'rca_%s_%s'%(size, split)
    #    #        for size in all_sizes
    #    #        for split in all_splits
    #    #    ]
    #    #}
    #    #for size in all_sizes:
    #    #    splits['%s_all'%size] = {
    #    #        'shards': ['rca_%s_%s'%(size, split) for split in all_splits]
    #    #    }
    #    #    
    #    #    for split in all_splits:
    #    #        splits['%s_%s'%(size, split)] = {
    #    #            'shards':['rca_%s_%s'%(size, split)]
    #    #        }
    #    
    #    build_dataset_info(
    #        args.dataset,
    #        module.brick_shapes,
    #        module.brick_colors,
    #        module.max_instances_per_scene,
    #        module.max_edges_per_scene,
    #        splits={},
    #    )
