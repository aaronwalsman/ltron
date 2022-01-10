import numpy

import ltron.utils as utils
import ltron.gym.spaces as bg_spaces
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class InstanceListComponent(LtronGymComponent):
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
        
        self.observation_space = bg_spaces.ClassLabelSpace(
                self.num_classes, self.max_instances)
        
    def compute_observation(self):
        brick_scene = self.scene_component.brick_scene
        instance_labels = numpy.zeros(
                (self.max_instances+1, 1), dtype=numpy.long)
        for instance_id, instance in brick_scene.instances.items():
            if self.filter_hidden and brick_scene.instance_hidden(instance):
                continue
            brick_shape_name = str(instance.brick_shape)
            shape_id = self.dataset_component.get_shape_id(brick_shape_name)
            instance_labels[instance_id, 0] = shape_id
        self.observation = instance_labels
    
    def reset(self):
        self.compute_observation()
        return self.observation
    
    def step(self, action):
        self.compute_observation()
        return self.observation, 0., False, None

class InstanceGraphComponent(LtronGymComponent):
    def __init__(self,
            num_classes,
            max_instances,
            max_snaps,
            max_edges,
            dataset_component,
            scene_component):
        self.num_classes = num_classes
        self.max_instances = max_instances
        self.max_snaps = max_snaps
        self.max_edges = max_edges
        self.dataset_component = dataset_component
        self.scene_component = scene_component
        self.scene_component.brick_scene.make_track_snaps()
        
        self.observation_space = bg_spaces.InstanceGraphSpace(
            self.num_classes,
            self.max_instances,
            self.max_snaps,
            self.max_edges,
        )
    
    def compute_observation(self):
        brick_scene = self.scene_component.brick_scene
        return self.observation_space.from_scene(
            brick_scene,
            self.dataset_component.dataset_info['shape_ids'],
        )
        '''
        snap_connections = brick_scene.get_all_snap_connections()
        unidirectional_edges = set()
        edge_index = 0
        for instance_name in snap_connections:
            instance_id = int(instance_name)
            for other_name, snap_a, snap_b in snap_connections[instance_name]:
                other_id = int(other_name)
                if other_id < instance_id:
                    unidirectional_edges.add((other_id, instance_id))
                else:
                    unidirectional_edges.add((instance_id, other_id))
        
        edge_index = numpy.zeros((2, self.max_edges), dtype=numpy.long)
        for i, edge in enumerate(unidirectional_edges):
            edge_index[:,i] = edge
        edge_data = {
            'edge_index' : edge_index,
            'num_edges' : len(unidirectional_edges),
        }
        
        return {
            'instances' : self.instance_list_component.compute_observation(),
            'edges' : edge_data,
        }
        '''
    
    def reset(self):
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None
