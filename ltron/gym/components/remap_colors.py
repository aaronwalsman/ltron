import numpy

from gymnasium.spaces import Tuple, Discrete

from supermecha import SuperMechaComponent

from ltron.constants import NUM_COLOR_CLASSES

class RandomRemapColorComponent(SuperMechaComponent):
    def __init__(self,
        initial_assembly_component,
        target_assembly_component,
        color_choices=(2,5,8,15,23,26),
    ):
        self.observation_space = Tuple([
            Discrete(NUM_COLOR_CLASSES),
            Discrete(NUM_COLOR_CLASSES),
        ])
        self.initial_assembly_component = initial_assembly_component
        self.target_assembly_component = target_assembly_component
        self.color_choices = color_choices
    
    def make_observation(self):
        return numpy.array([self.color_to_change, self.target_color], dtype=int)
    
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        initial_assembly = self.initial_assembly_component.observation
        colors = list(numpy.unique(initial_assembly['color']))
        colors.remove(0)
        if len(colors):
            self.color_to_change = self.np_random.choice(colors)
            self.target_color = self.np_random.choice(self.color_choices)
            color_match = initial_assembly['color'] == self.color_to_change
            initial_assembly['color'][color_match] = self.target_color
        else:
            self.color_to_change = 0
            self.target_color = 0
        return self.make_observation(), {}
    
    def step(self, action):
        for assembly in self.target_assembly_component.observations:
            color_match = (assembly['color'] == self.color_to_change)
            assembly['color'][color_match] = self.target_color
        
        return self.make_observation(), 0., False, False, {}
