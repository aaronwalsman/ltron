import numpy

import gym.spaces as gym_spaces

import ltron.gym.spaces as bg_spaces
from ltron.gym.components.brick_env_component import BrickEnvComponent

class BrickPosition(BrickEnvComponent):
    def __init__(self,
            max_instances_per_scene,
            scene_component):
        
        self.max_instances_per_scene = max_instances_per_scene
        self.scene_component = scene_component
        
        self.observation_space = gym_spaces.Dict({
                'world' : gym_spaces.Box(
                    -1000, 1000, shape=(self.max_instances_per_scene+1,3)),
                'camera' : gym_spaces.Box(
                    -1000, 1000, shape=(self.max_instances_per_scene+1,3))})
    
    def compute_observation(self):
        scene = self.scene_component.brick_scene
        world_observation = numpy.zeros((self.max_instances_per_scene+1, 3))
        camera_observation = numpy.zeros((self.max_instances_per_scene+1, 3))
        camera_inv_pose = scene.get_camera_pose()
        for instance_id, instance in scene.instances.items():
            if not scene.instance_hidden(instance_id):
                world_observation[instance_id, 0] = instance.transform[0,3]
                world_observation[instance_id, 1] = instance.transform[1,3]
                world_observation[instance_id, 2] = instance.transform[2,3]
                relative_transform = camera_inv_pose @ instance.transform
                camera_observation[instance_id, 0] = relative_transform[0,3]
                camera_observation[instance_id, 1] = relative_transform[1,3]
                camera_observation[instance_id, 2] = relative_transform[2,3]
        
        return {'world' : world_observation, 'camera' : camera_observation}
    
    def reset(self):
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None
