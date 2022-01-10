import random
import json

import numpy

from pyquaternion import Quaternion

from ltron.geometry.utils import squared_distance
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class RandomFloatingPairs(LtronGymComponent):
    def __init__(self,
            scene_component,
            augmentations,
            colors,
            pairs_per_scene = (10,20),
            min_brick_distance = 50,
            brick_placement_distance = (50, 100),
            rotation_mode = 'identity'):
        
        self.scene_component = scene_component
        if isinstance(augmentations, str):
            augmentations = json.load(open(augmentations))
        self.augmentations = augmentations
        self.colors = colors
        self.pairs_per_scene = pairs_per_scene
        self.min_brick_distance = min_brick_distance
        self.brick_placement_distance = brick_placement_distance
        self.rotation_mode = rotation_mode
        
    def reset(self):
        try:
            len(self.pairs_per_scene)
            pairs_per_scene = self.pairs_per_scene
        except TypeError:
            pairs_per_scene = (self.pairs_per_scene, self.pairs_per_scene)
        num_new_pairs = random.randint(*pairs_per_scene)
        
        try:
            len(self.brick_placement_distance)
            brick_placement_distance = self.brick_placement_distance
        except TypeError:
            brick_placement_distance = (
                    self.brick_placement_distance,
                    self.brick_placement_distance)
        
        brick_scene = self.scene_component.brick_scene
        
        for brick in range(num_new_pairs):
            
            all_brick_transforms = [instance.transform
                    for instance in brick_scene.instances.values()]
            all_brick_positions = [numpy.dot(transform, [0,0,0,1])[:3]
                    for transform in all_brick_transforms]
            
            if len(all_brick_positions):
                while True:
                    distance = random.uniform(*brick_placement_distance)
                    offset = [1,1,1]
                    while squared_distance(offset, (0,0,0)) > 1.:
                        offset = [random.random() * 2 - 1. for _ in range(3)]
                    norm = squared_distance(offset, (0,0,0))**0.5
                    offset = [o*distance/norm for o in offset]
                    
                    seed_id = random.randint(0, len(all_brick_transforms)-1)
                    seed_transform = all_brick_transforms[seed_id]
                    
                    offset_position = numpy.dot(
                            seed_transform, offset + [1])[:3]
                    
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
                        if self.rotation_mode == 'identity':
                            new_brick_transform[:3,:3] = numpy.eye(3)
                        elif self.rotation_mode == 'uniform':
                            q = Quaternion.random()
                            new_brick_transform[:3,:3] = q.rotation_matrix
                        break
            
            else:
                if (self.rotation_mode == 'identity' or
                        self.rotation_mode == 'local_identity'):
                    new_brick_transform = numpy.array([
                            [1, 0, 0, 0],
                            [0,-1, 0, 0],
                            [0, 0,-1, 0],
                            [0, 0, 0, 1]])
                elif self.rotation_mode == 'uniform':
                    q = Quaternion.random()
                    new_brick_transform = q.transformation_matrix
            brick_shape_a = random.choice(list(self.augmentations.keys()))
            brick_shape_b = random.choice(
                    list(self.augmentations[brick_shape_a].keys()))
            offset_transform = random.choice(
                    self.augmentations[brick_shape_a][brick_shape_b])
            
            for brick_shape in brick_shape_a, brick_shape_b:
                if brick_shape not in brick_scene.shape_library:
                    brick_scene.add_brick_shape(brick_shape)
            
            brick_color_a = random.choice(self.colors)
            brick_color_b = random.choice(self.colors)
            for brick_color in brick_color_a, brick_color_b:
                if brick_color not in brick_scene.color_library:
                    brick_scene.load_colors([brick_color])
            
            brick_scene.add_instance(
                    brick_shape_a, brick_color_b, new_brick_transform)
            brick_scene.add_instance(
                    brick_shape_b,
                    brick_color_b,
                    numpy.dot(new_brick_transform, offset_transform))
        
        return None
