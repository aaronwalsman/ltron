from gym.spaces import Dict

from ltron.bricks.brick_scene import TooManyInstancesError, make_empty_assembly
from ltron.gym.spaces import AssemblySpace
from ltron.gym.components.sensor_component import SensorComponent

class AssemblyComponent(SensorComponent):
    def __init__(self,
        scene_component,
        shape_ids,
        color_ids,
        max_instances,
        max_edges,
        update_frequency,
        observable,
    ):
        super().__init__(
            update_frequency=update_frequency,
            observable=observable,
        )
        
        self.scene_component = scene_component
        self.shape_ids = shape_ids
        self.color_ids = color_ids
        self.max_instances = max_instances
        self.max_edges = max_edges
        
        if self.observable:
            self.observation_space = AssemblySpace(
                self.shape_ids,
                self.color_ids,
                self.max_instances,
                self.max_edges,
            )
    
    def update_observation(self):
        self.observation = self.scene_component.brick_scene.get_assembly(
            self.shape_ids,
            self.color_ids,
            self.max_instances,
            self.max_edges,
        )
    
    '''
    def update_observation(self):
        #try:
        self.observation = self.scene_component.brick_scene.get_assembly(
            self.shape_ids,
            self.color_ids,
            self.max_instances,
            self.max_edges,
        )
        #except TooManyInstancesError:
        #    self.terminal = True
        #    self.assembly = make_empty_assembly(
        #        self.max_instances, self.max_edges)
        
    '''
