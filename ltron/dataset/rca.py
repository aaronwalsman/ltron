from ltron.sampler.subassembly_sampler import SingleSubAssemblySampler
from ltron.sampler.scene_sampler import sample_dataset

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

min_max_bricks_per_scene = [(2,2), (3,4), (5,8), (9,16), (17,32)]

def build_rca():
    sample_dataset(
        'rca',
        subassembly_samplers,
        colors,
        min_max_bricks_per_scene,
        train_scenes = 50000,
        test_scenes = 10000,
    )
