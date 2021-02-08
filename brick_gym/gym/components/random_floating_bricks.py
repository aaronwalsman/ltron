import random

import numpy

from brick_gym.geometry.utils import squared_distance
from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class RandomFloatingBricks(BrickEnvComponent):
    def __init__(self,
            scene_component,
            bricks,
            colors,
            bricks_per_scene = (10,20),
            min_brick_distance = 50,
            brick_placement_distance = (50, 100)):
        
        self.scene_component = scene_component
        self.bricks = bricks
        self.colors = colors
        self.bricks_per_scene = bricks_per_scene
        self.min_brick_distance = min_brick_distance
        self.brick_placement_distance = brick_placement_distance
        
    def reset(self):
        try:
            len(self.bricks_per_scene)
            bricks_per_scene = self.bricks_per_scene
        except TypeError:
            bricks_per_scene = (self.bricks_per_scene, self.bricks_per_scene)
        num_new_bricks = random.randint(*bricks_per_scene)
        
        try:
            len(self.brick_placement_distance)
            brick_placement_distance = self.brick_placement_distance
        except TypeError:
            brick_placement_distance = (
                    self.brick_placement_distance,
                    self.brick_placement_distance)
        
        brick_scene = self.scene_component.brick_scene
        
        for brick in range(num_new_bricks):
            
            all_brick_transforms = [instance.transform
                    for instance in brick_scene.instances.values()]
            all_brick_positions = [numpy.dot(transform, [0,0,0,1])[:3]
                    for transform in all_brick_transforms]
            
            while True:
                distance = random.uniform(*brick_placement_distance)
                offset = [1,1,1]
                while squared_distance(offset, (0,0,0)) > 1.:
                    offset = [random.random() * 2 - 1. for _ in range(3)]
                norm = squared_distance(offset, (0,0,0))**0.5
                offset = [o*distance/norm for o in offset]
                
                seed_id = random.randint(0, len(all_brick_transforms)-1)
                seed_transform = all_brick_transforms[seed_id]
                
                offset_position = numpy.dot(seed_transform, offset + [1])[:3]
                
                far_enough = [
                        squared_distance(position, offset_position) >
                            self.min_brick_distance**2
                        for position in all_brick_positions]
                if all(far_enough):
                    new_brick_transform = numpy.dot(seed_transform,
                            numpy.array([
                                [1, 0, 0, offset[0]],
                                [0, 1, 0, offset[1]],
                                [0, 0, 1, offset[2]],
                                [0, 0, 0, 1]]))
                    break
            
            brick_type = random.choice(self.bricks)
            if brick_type not in brick_scene.brick_library:
                brick_scene.brick_library.add_type(brick_type)
            brick_color = random.choice(self.colors)
            if brick_color not in brick_scene.color_library:
                brick_scene.load_colors([brick_color])
            brick_scene.add_instance(
                    brick_type, brick_color, new_brick_transform)
        
        return None
