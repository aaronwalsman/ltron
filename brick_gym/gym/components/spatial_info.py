import numpy

import gym.spaces as gym_spaces

import brick_gym.gym.spaces as bg_spaces
from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class BrickHeight(BrickEnvComponent):
    def __init__(self,
            max_instances_per_scene,
            scene_component):
        
        self.max_instances_per_scene = max_instances_per_scene
        self.scene_component = scene_component
        
        self.observation_space = gym_spaces.Box(
                -1000, 1000, shape=(self.max_instances_per_scene+1,))
    
    def compute_observation(self):
        scene = self.scene_component.brick_scene
        observation = numpy.zeros(self.max_instances_per_scene+1)
        for instance_id, instance in scene.instances.items():
            y = instance.transform[1,3]
            if not scene.instance_hidden(instance_id):
                observation[instance_id] = y
        
        return observation
    
    def reset(self):
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None
