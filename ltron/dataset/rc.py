from argparse import ArgumentParser

import ltron.dataset.rca as rca

modules = {
    'rca':rca,
}

parser = ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--size', type=str, nargs='*', default='all')
parser.add_argument('--split', type=str, nargs='*', default='all')

known_datasets = {
    'rca', 'rcb', 'rcc'
}

def build_rc_dataset():
    args = parser.parse_args()
    
    if args.size == 'all':
        sizes = modules[args.dataset].all_sizes
    else:
        sizes = args.size
    
    if args.split == 'all':
        splits = modules[args.dataset].all_splits
    else:
        splits = args.splits
    
    collections = []
    for size in sizes:
        min_instances, max_instances = size.split('_')
        min_instances = int(min_instances)
        max_instances = int(max_instances)
        for split in splits:
            collections.append((min_instances, max_instances, split))
    
    if args.dataset == 'rca':
        rca.build(collections)
    elif args.dataset == 'rcb':
        raise NotImplementedError
    elif args.dataset == 'rcc':
        raise NotImplementedError
    else:
        raise ValueError('dataset argument must be one of %s'%known_datasets)
