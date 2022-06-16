from ltron.dataset.sampler.subassembly_sampler import SingleSubAssemblySampler
from ltron.dataset.sampler.scene_sampler import sample_collection

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
    SingleSubAssemblySampler('2436.dat'),
    SingleSubAssemblySampler('4081.dat'),
]

colors = ['1','4','7','14','22','25']

all_sizes = ['2_2', '3_4', '5_8', '9_16', '17_32']
all_splits = ['train', 'test']

collection_sizes = {
    '2_2_train' : 500000,
    '2_2_test' : 10000,
    '3_4_train' : 50000,
    '3_4_test' : 10000,
    '5_8_train' : 50000,
    '5_8_test' : 10000,
    '9_16_train' : 50000,
    '9_16_test' : 10000,
    '17_32_train' : 50000,
    '17_32_test' : 10000,
}

def build(collections):
    for collection in collections:
        min_instances, max_instances, split = collection
        collection_name = '%i_%i_%s'%(min_instances, max_instances, split)
        if collection_name not in collection_sizes:
            raise ValueError('Invalid rca collection: "%s"'%collection_name)
        sample_collection(
            'rca',
            split,
            subassembly_samplers,
            colors,
            min_instances,
            max_instances,
            collection_sizes[collection_name],
        )
