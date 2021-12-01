from gym.spaces import Dict

from ltron.gym.spaces import ConfigurationSpace
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class ConfigComponent(LtronGymComponent):
    def __init__(self,
        scene_component,
        class_ids,
        color_ids,
        max_instances,
        max_edges,
        update_frequency,
        observe_config,
    ):
        self.scene_component = scene_component
        self.class_ids = class_ids
        self.color_ids = color_ids
        self.max_instances = max_instances
        self.max_edges = max_edges
        self.update_frequency = update_frequency
        self.observe_config = observe_config
        
        if self.observe_config:
            self.observation_space = Dict({
                'config' : ConfigurationSpace(
                    self.class_ids,
                    self.color_ids,
                    self.max_instances,
                    self.max_edges,
                ),
            })
        
    def observe(self, initial=False):
        if self.update_frequency == 'step' or initial:
            self.config = self.scene_component.brick_scene.get_configuration(
                self.class_ids,
                self.color_ids,
                self.max_instances,
                self.max_edges,
            )
        
        if self.observe_config:
            self.observation = {
                'config' : self.config,
            }
        else:
            self.observation = None
    
    def reset(self):
        self.observe(initial=True)
        return self.observation
    
    def step(self, action):
        self.observe()
        return self.observation, 0., False, None
    
    def set_state(self, state):
        self.config = state
        self.observe()
        return self.observation
    
    def get_state(self):
        return self.config
