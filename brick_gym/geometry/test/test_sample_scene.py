#!/usr/bin/env python
from brick_gym.geometry.scene_sampler import (
        sample_scene, SingleSubAssemblySampler)

samplers = [
        SingleSubAssemblySampler('3003.dat'),
        SingleSubAssemblySampler('3001.dat'),
        SingleSubAssemblySampler('2436a.dat'),
]

scene = sample_scene(samplers, 10, [1,4], debug=True)
scene.export_ldraw('./tmp.ldr')
