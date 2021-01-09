import numpy

import brick_gym.utils as utils
import brick_gym.gym.spaces as bg_spaces
from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class InstanceLabelComponent(BrickEnvComponent):
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
        
        self.observation_space = bg_spaces.NodeSpace(
                self.num_classes, self.max_instances+1)
        
    def compute_observation(self):
        brick_scene = self.scene_component.brick_scene
        observation = numpy.zeros(self.max_instances+1, dtype=numpy.long)
        for instance_id, instance in brick_scene.instances.items():
            if self.filter_hidden and brick_scene.instance_hidden(instance):
                continue
            brick_type_name = str(instance.brick_type)
            class_id = self.dataset_component.get_class_id(brick_type_name)
            observation[instance_id] = class_id
        
        return observation
    
    def reset(self):
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None

class GraphLabelComponent(BrickEnvComponent):
    def __init__(self,
            num_classes,
            max_nodes,
            scene_component):
        self.num_classes=num_classes
        self.max_nodes=max_nodes
        self.scene_component = scene_component
        self.scene_component.brick_gym.make_track_snaps()
        
        self.observation_space = bg_spaces.GraphScoreSpace(
                self.num_classes, self.max_nodes)
    
    def compute_observation(self):
        brick_scene = self.scene_component.brick_scene
        SOME_NEW_THING
        #node_labels, edge_labels = utils.metadata_to_graph(
        #        metadata, max_nodes=self.max_nodes)
        observation = {
                'nodes' : node_labels,
                'edges' : edge_labels}
        return observation
    
    def reset(self):
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None
