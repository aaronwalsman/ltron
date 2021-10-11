import random

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class RandomizeColorsComponent(LtronGymComponent):
    def __init__(self,
            color_ids,
            scene_component,
            randomize_frequency='step'):
        
        self.color_ids = color_ids
        self.scene_component = scene_component
        assert randomize_frequency in ('step', 'reset')
        self.randomize_frequency = randomize_frequency
        
        scene_component.brick_scene.load_colors(self.color_ids.values())
    
    def randomize_colors(self):
        #randomized_colors = list(self.all_colors)
        #random.shuffle(randomized_colors)
        randomized_colors = list(self.color_ids.keys())
        random.shuffle(randomized_colors)
        color_mapping = dict(zip(
            self.color_ids.keys(), randomized_colors))
        for instance in self.scene_component.brick_scene.instances.values():
            new_color = color_mapping[instance.color.color_name]
            self.scene_component.brick_scene.set_instance_color(
                    instance, new_color)
    
    def reset(self):
        self.randomize_colors()
    
    def step(self, action):
        if self.randomize_frequency == 'step':
            self.randomize_colors()
        return None, 0., False, None
    
    #def set_state(self, state):
    #    self.randomize_colors()
