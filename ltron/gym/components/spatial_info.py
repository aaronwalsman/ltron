import numpy

import gym.spaces as gym_spaces

import ltron.gym.spaces as ltron_spaces
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class BrickPosition(LtronGymComponent):
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
        view_matrix = scene.get_view_matrix()
        for instance_id, instance in scene.instances.items():
            if not scene.instance_hidden(instance_id):
                world_observation[instance_id, 0] = instance.transform[0,3]
                world_observation[instance_id, 1] = instance.transform[1,3]
                world_observation[instance_id, 2] = instance.transform[2,3]
                relative_transform = view_matrix @ instance.transform
                camera_observation[instance_id, 0] = relative_transform[0,3]
                camera_observation[instance_id, 1] = relative_transform[1,3]
                camera_observation[instance_id, 2] = relative_transform[2,3]
        
        return {'world' : world_observation, 'camera' : camera_observation}
    
    def reset(self):
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None

class InstancePoseComponent(LtronGymComponent):
    def __init__(self,
        max_instances_per_scene,
        scene_component,
        space='camera',
        scene_min=-1000,
        scene_max=1000,
    ):
        
        self.max_instances_per_scene = max_instances_per_scene
        self.scene_component = scene_component
        self.space = space
        
        self.observation_space = ltron_spaces.MultiSE3Space(
            self.max_instances_per_scene,
            scene_min,
            scene_max,
        )
    
    def compute_observation(self):
        scene = self.scene_component.brick_scene
        self.observation = numpy.zeros((self.max_instances_per_scene+1, 4, 4))
        self.observation[0] = numpy.eye(4)
        for instance_id, instance in scene.instances.items():
            if self.space == 'world':
                transform_offset = numpy.eye(4)
            elif self.space == 'camera':
                transform_offset = scene.get_view_matrix()
            else:
                raise NotImplementedError
            self.observation[instance_id] = (
                transform_offset @ instance.transform)
    
    def reset(self):
        self.compute_observation()
        return self.observation
    
    def step(self, action):
        self.compute_observation()
        return self.observation, 0., False, None
