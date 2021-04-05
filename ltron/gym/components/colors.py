import random

from ltron.gym.components.brick_env_component import BrickEnvComponent

class RandomizeColorsComponent(BrickEnvComponent):
    def __init__(self,
            all_colors,
            scene_component,
            randomize_frequency='step'):
        
        self.all_colors = all_colors
        self.scene_component = scene_component
        self.randomize_frequency = randomize_frequency
        
        scene_component.brick_scene.load_colors(all_colors)
    
    def randomize_colors(self):
        randomized_colors = list(self.all_colors)
        random.shuffle(randomized_colors)
        color_mapping = dict(zip(self.all_colors, randomized_colors))
        for instance in self.scene_component.brick_scene.instances.values():
            new_color = color_mapping[instance.color]
            self.scene_component.brick_scene.set_instance_color(
                    instance, new_color)
    
    def reset(self):
        self.randomize_colors()
    
    def step(self, action):
        if self.randomize_frequency == 'step':
            self.randomize_colors()
        return None, 0., False, None
    
    def set_state(self, state):
        self.randomize_colors()
