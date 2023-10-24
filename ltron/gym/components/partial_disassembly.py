import random

import numpy

from supermecha import SuperMechaComponent

class PartialDisassemblyComponent(SuperMechaComponent):
    def __init__(self, scene_component, min_remaining):
        self.scene_component = scene_component
        self.min_remaining = min_remaining
    
    def reset(self, seed=None, options=None):
        scene = self.scene_component.brick_scene
        heights = [
            (scene.instances[i].transform[1,3], i)
            for i in scene.instances
        ]
        sorted_instances = [i[1] for i in sorted(heights, reverse=True)]
        num_instances = len(scene.instances)
        high = max(2, len(scene.instances)-self.min_remaining)
        num_remove = self.np_random.integers(low=1, high=high)
        
        for i in range(num_remove):
            scene.remove_instance(sorted_instances[i])
        
        return None, {}
