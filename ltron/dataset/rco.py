from ltron.dataset.sampler.subassembly_sampler import (
    SingleSubAssemblySampler,
    get_all_brick_shapes,
)
from ltron.dataset.sampler.scene_sampler import sample_shard

subassembly_samplers = [
    SingleSubAssemblySampler('54383.dat'),
    SingleSubAssemblySampler('41770.dat'),
    SingleSubAssemblySampler('2450.dat'),
    SingleSubAssemblySampler('43722.dat'),
    SingleSubAssemblySampler('2436.dat'),
    SingleSubAssemblySampler('4081.dat'),
]

brick_shapes = get_all_brick_shapes(subassembly_samplers)
brick_colors = ['1','4','7','14','22','25']

all_sizes = ['2_2', '4_4', '8_8']
all_splits = ['train', 'test']
shard_sizes = {
    '2_2_train' : 50000,
    '2_2_test' : 1000,
    '4_4_train' : 50000,
    '4_4_test' : 1000,
    '8_8_train' : 50000,
    '8_8_test' : 1000,
}

max_instances_per_scene = 32
max_edges_per_scene = 512

def build(shards):
    for shard in shards:
        min_instances, max_instances, split = shard
        shard_name = '%i_%i_%s'%(min_instances, max_instances, split)
        if shard_name not in shard_sizes:
            raise ValueError('Invalid rco shard: "%s"'%shard_name)
        sample_shard(
            'rco',
            split,
            subassembly_samplers,
            brick_colors,
            min_instances,
            max_instances,
            shard_sizes[shard_name],
            place_on_top=True
        )
