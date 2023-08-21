from ltron.dataset.sampler.subassembly_sampler import (
    SingleSubAssemblySampler,
    get_all_brick_shapes,
)
from ltron.dataset.sampler.scene_sampler import sample_shard

'''
The rca dataset contains six brick types and six colors and is designed to be
the easiest of all randomly generated datsets.  All bricks are rotationally
assymetric with no complex subassemblies.
'''

subassembly_samplers = [
    SingleSubAssemblySampler('54383.dat'),
    SingleSubAssemblySampler('41770.dat'),
    SingleSubAssemblySampler('2450.dat'),
    SingleSubAssemblySampler('43722.dat'),
    #SingleSubAssemblySampler('2436.dat'),
    #SingleSubAssemblySampler('4081.dat'),
]

brick_shapes = get_all_brick_shapes(subassembly_samplers)
brick_colors = ['1','4','7','14','22','25']

all_sizes = ['9_16', '17_32'] #['2_2'] #, '3_4', '5_8', '9_16', '17_32']
all_splits = ['train', 'test']

split_sizes = {
    '2_2_train' : 50000,
    '2_2_test' : 10000,
    '3_4_train' : 50000,
    '3_4_test' : 10000,
    '5_8_train' : 50000,
    '5_8_test' : 10000,
    '9_16_train' : 200000,
    '9_16_test' : 10000,
    '17_32_train' : (25000,25000,25000,25000,25000,25000,25000,25000),#200000,
    '17_32_test' : 50000,
}

max_instances_per_scene = 32
max_edges_per_scene = 512

def build(shards):
    for shard in shards:
        min_instances, max_instances, split, *shard_ids = shard
        if len(shard_ids):
            for shard_id in shard_ids:
                split_name = '%i_%i_%s'%(min_instances, max_instances, split)
                shard_name = '%i_%i_%s_%i'%(
                    min_instances, max_instances, split, shard_id)
                if split_name not in split_sizes:
                    raise ValueError('Invalid rcb split: "%s"'%split_name)
                sample_shard(
                    'rcb',
                    split,
                    subassembly_samplers,
                    brick_colors,
                    min_instances,
                    max_instances,
                    split_sizes[split_name][shard_id],
                    shard_id=shard_id,
                )
        else:
            shard_name = '%i_%i_%s'%(min_instances, max_instances, split)
            if shard_name not in split_sizes:
                raise ValueError('Invalid rcb shard: "%s"'%shard_name)
            sample_shard(
                'rcb',
                split,
                subassembly_samplers,
                brick_colors,
                min_instances,
                max_instances,
                split_sizes[shard_name],
            )
