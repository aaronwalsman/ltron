import math
import time
from itertools import product

import numpy
random = numpy.random.default_rng(1234567890)

from pyquaternion import Quaternion

from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.snap import SnapCylinder
from ltron.geometry.collision_sampler import get_all_transformed_snap_pairs
from ltron.geometry.collision import check_snap_collision

def get_all_brick_shapes(subassembly_samplers):
    return list(set(sum(
        [sampler.brick_shapes for sampler in subassembly_samplers],
        []
    )))

class SubAssemblySampler:
    pass

class SingleSubAssemblySampler(SubAssemblySampler):
    def __init__(self, brick_shape, transform=BrickScene.upright):
        self.brick_shapes = [brick_shape]
        self.transform = transform
    
    def sample(self):
        return [(self.brick_shapes[0], self.transform)]

class MultiSubAssemblySampler(SubAssemblySampler):
    def __init__(self,
        brick_shapes,
        transforms,
        global_transform=BrickScene.upright
    ):
        self.brick_shapes = brick_shapes
        self.transforms = transforms
        self.global_transform = global_transform
    
    def sample(self):
        transforms = [self.global_transform @ t for t in self.transforms]
        return list(zip(self.brick_shapes, transforms))

class RotatorSubAssemblySampler(SubAssemblySampler):
    def __init__(self,
        holder_type,
        rotor_type,
        axis,
        rotor_range,
        pivot=(0,0,0),
        transform=BrickScene.upright
    ):
        self.holder_type = holder_type
        self.rotor_type = rotor_type
        self.axis = axis
        self.rotor_range = rotor_range
        self.pivot = numpy.array(pivot)
        self.transform = transform
        self.brick_shapes = [holder_type, rotor_type]
    
    def sample(self):
        theta = random.uniform(*self.rotor_range)
        r = Quaternion(axis=self.axis, angle=theta).transformation_matrix
        pivot_a = numpy.eye(4)
        pivot_a[:3,3] = -self.pivot
        pivot_b = numpy.eye(4)
        pivot_b[:3,3] = self.pivot
        rotor_transform = self.transform @ pivot_b @ r @ pivot_a
        holder_transform = self.transform
        return [(self.holder_type, holder_transform),
                (self.rotor_type, rotor_transform)]

class AxleWheelSubAssemblySampler(SubAssemblySampler):
    def __init__(self,
        axle_type,
        wheel_transforms,
        wheel_samplers,
        transform=BrickScene.upright
    ):
        self.axle_type = axle_type
        self.wheel_transforms = wheel_transforms
        self.wheel_samplers = wheel_samplers
        self.transform = transform
        self.brick_shapes = [axle_type] + get_all_brick_shapes(wheel_samplers)
    
    def sample(self):
        axle_transform = self.transform
        wheel_sampler = random.choice(self.wheel_samplers)
        wheel_instances = []
        for wheel_transform in self.wheel_transforms:
            instances = wheel_sampler.sample()
            types, transforms = zip(*instances)
            transforms = [
                    self.transform @ wheel_transform @ t for t in transforms]
            wheel_instances.extend(zip(types, transforms))
        
        return [(self.axle_type, axle_transform)] + wheel_instances
