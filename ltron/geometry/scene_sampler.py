import math
import time
from itertools import product

import numpy
random = numpy.random.default_rng(1234567890)

from pyquaternion import Quaternion

import splendor.contexts.egl as egl
from splendor.frame_buffer import FrameBufferWrapper

from ltron.dataset.paths import get_dataset_info
from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.snap import SnapCylinder
from ltron.geometry.collision_sampler import get_all_transformed_snap_pairs
from ltron.geometry.collision import check_snap_collision

class SubAssemblySampler:
    pass

class SingleSubAssemblySampler(SubAssemblySampler):
    def __init__(self, brick_shape, transform=BrickScene.upright):
        self.brick_shape = brick_shape
        self.transform = transform
    
    def sample(self):
        return [(self.brick_shape, self.transform)]

class MultiSubAssemblySampler(SubAssemblySampler):
    def __init__(self,
            brick_shapes,
            transforms,
            global_transform=BrickScene.upright):
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
            transform=BrickScene.upright):
        self.holder_type = holder_type
        self.rotor_type = rotor_type
        self.axis = axis
        self.rotor_range = rotor_range
        self.pivot = numpy.array(pivot)
        self.transform = transform
    
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
            transform=BrickScene.upright):
        self.axle_type = axle_type
        self.wheel_transforms = wheel_transforms
        self.wheel_samplers = wheel_samplers
        self.transform = transform
    
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

AntennaSampler = RotatorSubAssemblySampler(
        '4592.dat', '4593.dat', (1,0,0), (-math.pi/2, math.pi/2))

SpinnerPlateSampler = RotatorSubAssemblySampler(
        '3680.dat', '3679.dat', (0,1,0), (0, math.pi*2))

FolderSampler = RotatorSubAssemblySampler(
        '3937.dat', '3938.dat', (1,0,0), (0, math.pi/2), pivot=(0,10,0))

WheelASampler = MultiSubAssemblySampler(
        ['30027b.dat', '30028.dat'],
        [numpy.array([
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1,-2],
            [ 0, 0, 0, 1]]),
         numpy.array([
            [ 0, 0, 1, 0],
            [ 0, 1, 0, 0],
            [-1, 0, 0, 3],
            [ 0, 0, 0, 1]])],
        global_transform = numpy.eye(4))

WheelBSampler = MultiSubAssemblySampler(
        ['4624.dat', '3641.dat'],
        [numpy.eye(4), numpy.eye(4)],
        global_transform = numpy.eye(4))

WheelCSampler = MultiSubAssemblySampler(
        ['6014.dat', '6015.dat'],
        [numpy.eye(4), numpy.array([
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1,-6],
            [ 0, 0, 0, 1]])],
        global_transform = numpy.eye(4))

RegularAxleWheelSampler = AxleWheelSubAssemblySampler(
        '4600.dat',
        [numpy.array([
            [0, 0,-1, 30],
            [0, 1, 0,  5],
            [1, 0, 0,  0],
            [0, 0, 0,  1]]),
         numpy.array([
            [0, 0, 1,-30],
            [0, 1, 0,  5],
            [1, 0, 0,  0],
            [0, 0, 0,  1]])],
        [WheelASampler, WheelBSampler, WheelCSampler])

WideAxleWheelSampler = AxleWheelSubAssemblySampler(
        '6157.dat',
        [numpy.array([
            [0, 0,-1, 40],
            [0, 1, 0,  5],
            [1, 0, 0,  0],
            [0, 0, 0,  1]]),
         numpy.array([
            [0, 0, 1,-40],
            [0, 1, 0,  5],
            [1, 0, 0,  0],
            [0, 0, 0,  1]])],
        [WheelASampler, WheelBSampler, WheelCSampler])
        
def get_all_snap_pairs(instance_snaps_a, instance_snaps_b):
    snap_pairs = [
        (snap_a, snap_b)
        for (snap_a, snap_b)
        in product(instance_snaps_a, instance_snaps_b)
        if (snap_a != snap_b and snap_a.compatible(snap_b))
    ]

    return snap_pairs

def sample_scene(
        scene,
        sub_assembly_samplers,
        parts_per_scene,
        colors,
        retries=20,
        collision_resolution=(512,512),
        debug=False,
        timeout=None):
    
    t_start = time.time()
    
    try:
        len(parts_per_scene)
    except TypeError:
        parts_per_scene = parts_per_scene, parts_per_scene
    
    num_parts = random.integers(*parts_per_scene, endpoint=True)
    
    scene.load_colors(colors)
    
    for i in range(num_parts):
        if timeout is not None:
            if time.time() - t_start > timeout:
                print('TIMEOUT')
                return scene
        
        if len(scene.instances):
            
            unoccupied_snaps = scene.get_unoccupied_snaps()
            
            if not len(unoccupied_snaps):
                print('no unoccupied snaps!')
            
            while True:
                if timeout is not None:
                    if time.time() - t_start > timeout:
                        print('TIMEOUT')
                        return scene
                #===============================================================
                # import the sub-assembly
                sub_assembly_sampler = random.choice(sub_assembly_samplers)
                sub_assembly = sub_assembly_sampler.sample()
                sub_assembly_snaps = []
                new_instances = []
                for brick_shape, transform in sub_assembly:
                    color = random.choice(colors)
                    scene.add_brick_shape(brick_shape)
                    new_instance = scene.add_instance(
                            brick_shape, color, transform)
                    new_instances.append(new_instance)
                    new_snaps = new_instance.snaps
                    sub_assembly_snaps.extend(new_snaps)
                
                # TMP to handle bad sub-assemblies
                if len(sub_assembly_snaps) == 0:
                    for instance in new_instances:
                        scene.remove_instance(instance)
                    continue
                
                #===============================================================
                # try to find a valid connection
                #---------------------------------------------------------------
                # get all pairs
                pairs = get_all_snap_pairs(sub_assembly_snaps, unoccupied_snaps)
                if len(pairs) == 0:
                    for instance in new_instances:
                        scene.remove_instance(instance)
                    continue
                
                #---------------------------------------------------------------
                # try to find a pair that is not in collision
                for j in range(retries):
                    pick, place = random.choice(pairs)
                    transforms = scene.all_pick_and_place_transforms(
                        pick, place, check_collision=True)
                    
                    if len(transforms):
                        transform = random.choice(transforms)
                        scene.move_instance(pick.brick_instance, transform)
                        
                        if debug:
                            print(i,j)
                            scene.export_ldraw('%i_%s.ldr'%(i,j))
                        
                        break
                
                #---------------------------------------------------------------
                # if we tried many times and didn't find a good connection,
                # loop back and try a new sub-assembly
                else:
                    for instance in new_instances:
                        scene.remove_instance(instance)
                    continue
                
                #---------------------------------------------------------------
                # if we did find a good connection, break out and move on
                # to the next piece
                break
                    
        else:
            while True:
                sub_assembly_sampler = random.choice(sub_assembly_samplers)
                sub_assembly = sub_assembly_sampler.sample()
                new_instances = []
                sub_assembly_snaps = []
                for brick_shape, transform in sub_assembly:
                    color = random.choice(colors)
                    scene.add_brick_shape(brick_shape)
                    new_instance = scene.add_instance(
                            brick_shape, color, transform)
                    new_instances.append(new_instance)
                    new_snaps = new_instance.snaps
                    sub_assembly_snaps.extend(new_snaps)
                
                if len(sub_assembly_snaps):
                    break
                else:
                    for instance in new_instances:
                        scene.remove_instance(instance)
    
    #return scene
