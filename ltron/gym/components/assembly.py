from gym.spaces import Dict

from ltron.bricks.brick_scene import TooManyInstancesError, make_empty_assembly
from ltron.gym.spaces import AssemblySpace
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class AssemblyComponent(LtronGymComponent):
    def __init__(self,
        scene_component,
        shape_ids,
        color_ids,
        max_instances,
        max_edges,
        update_frequency,
        observe_assembly,
    ):
        self.scene_component = scene_component
        self.shape_ids = shape_ids
        self.color_ids = color_ids
        self.max_instances = max_instances
        self.max_edges = max_edges
        self.update_frequency = update_frequency
        self.observe_assembly = observe_assembly
        
        if self.observe_assembly:
            self.observation_space = AssemblySpace(
                self.shape_ids,
                self.color_ids,
                self.max_instances,
                self.max_edges,
            )
        
    def observe(self, initial=False):
        if self.update_frequency == 'step' or initial:
            try:
                self.assembly = self.scene_component.brick_scene.get_assembly(
                    self.shape_ids,
                    self.color_ids,
                    self.max_instances,
                    self.max_edges,
                )
            except TooManyInstancesError:
                self.terminal = True
                self.assembly = make_empty_assembly(
                    self.max_instances, self.max_edges)
        
        if self.observe_assembly:
            self.observation = self.assembly
        else:
            self.observation = None
    
    def reset(self):
        self.terminal = False
        self.observe(initial=True)
        return self.observation
    
    def step(self, action):
        self.observe()
        return self.observation, 0., self.terminal, None
    
    def set_state(self, state):
        self.assembly = state
        self.observe()
        return self.observation
    
    def get_state(self):
        return self.assembly
