import numpy

import brick_gym.utils as utils
import brick_gym.gym.spaces as bg_spaces
from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class InstanceListComponent(BrickEnvComponent):
    def __init__(self,
            num_classes,
            max_instances,
            dataset_component,
            scene_component,
            filter_hidden = False):
        self.num_classes = num_classes
        self.max_instances = max_instances
        self.dataset_component = dataset_component
        self.scene_component = scene_component
        self.filter_hidden = filter_hidden
        
        self.observation_space = bg_spaces.InstanceListSpace(
                self.num_classes, self.max_instances)
        
    def compute_observation(self):
        brick_scene = self.scene_component.brick_scene
        observation = numpy.zeros((self.max_instances+1, 1), dtype=numpy.long)
        for instance_id, instance in brick_scene.instances.items():
            if self.filter_hidden and brick_scene.instance_hidden(instance):
                continue
            brick_type_name = str(instance.brick_type)
            class_id = self.dataset_component.get_class_id(brick_type_name)
            observation[instance_id, 0] = class_id
        
        return observation
    
    def reset(self):
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None

class InstanceGraphComponent(BrickEnvComponent):
    def __init__(self,
            num_classes,
            max_instances,
            max_edges,
            dataset_component,
            scene_component):
        self.num_classes = num_classes
        self.max_instances = max_instances
        self.max_edges = max_edges
        self.scene_component = scene_component
        self.scene_component.brick_scene.make_track_snaps()
        
        self.instance_list_component = InstanceListComponent(
                num_classes,
                max_instances,
                dataset_component,
                scene_component,
                filter_hidden=False)
        
        self.observation_space = bg_spaces.InstanceGraphSpace(
                self.num_classes, self.max_instances, self.max_edges)
    
    def compute_observation(self):
        brick_scene = self.scene_component.brick_scene
        snap_connections = brick_scene.get_all_snap_connections()
        unidirectional_edges = set()
        edge_index = 0
        for instance_name in snap_connections:
            instance_id = int(instance_name)
            for other_name, _ in snap_connections[instance_name]:
                other_id = int(other_name)
                if other_id < instance_id:
                    unidirectional_edges.add((other_id, instance_id))
                else:
                    unidirectional_edges.add((instance_id, other_id))
        
        edges = numpy.zeros((2, self.max_edges), dtype=numpy.long)
        for i, edge in enumerate(unidirectional_edges):
            edges[:,i] = edge
        return {
            'instances' : self.instance_list_component.compute_observation(),
            'edges' : edges,
        }
        '''
        return {
            'instances' :  numpy.zeros(self.max_instances+1, dtype=numpy.long),
            'edges' : numpy.zeros((2, self.max_edges), dtype=numpy.long)
        }
        '''
    
    def reset(self):
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None
