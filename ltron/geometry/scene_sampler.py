import random
import math
import time

import numpy

from pyquaternion import Quaternion

import splendor.contexts.egl as egl
from splendor.frame_buffer import FrameBufferWrapper

from ltron.dataset.paths import get_dataset_info
from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.snap import SnapCylinder
from ltron.geometry.collision_sampler import get_all_transformed_snap_pairs
from ltron.geometry.collision import check_collision

class SubAssemblySampler:
    pass

class SingleSubAssemblySampler(SubAssemblySampler):
    def __init__(self, brick_type, transform=BrickScene.upright):
        self.brick_type = brick_type
        self.transform = transform
    
    def sample(self):
        return [(self.brick_type, self.transform)]

class MultiSubAssemblySampler(SubAssemblySampler):
    def __init__(self,
            brick_types,
            transforms,
            global_transform=BrickScene.upright):
        self.brick_types = brick_types
        self.transforms = transforms
        self.global_transform = global_transform
    
    def sample(self):
        transforms = [self.global_transform @ t for t in self.transforms]
        return list(zip(self.brick_types, transforms))

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
        

def sample_scene(
        sub_assembly_samplers,
        parts_per_scene,
        colors,
        retries=20,
        collision_resolution=(512,512),
        debug=False,
        timeout=None):
    
    t_start = time.time()
    
    #manager = buffer_manager_egl.initialize_shared_buffer_manager()
    egl.initialize_plugin()
    egl.initialize_device()
    frame_buffer = FrameBufferWrapper(
            collision_resolution[0], collision_resolution[1], anti_alias=False)
    
    try:
        len(parts_per_scene)
    except TypeError:
        parts_per_scene = parts_per_scene, parts_per_scene
    
    num_parts = random.randint(*parts_per_scene)
    
    scene = BrickScene(renderable=True, track_snaps=True)
    scene.load_colors(colors)
    
    for i in range(num_parts):
        if timeout is not None:
            if time.time() - t_start > timeout:
                print('TIMEOUT')
                return scene
        
        if len(scene.instances):
            #===============================================================
            # get a list of unoccupied snaps
            #---------------------------------------------------------------
            # get all snaps
            all_scene_snaps = set()
            for instance_id, instance in scene.instances.items():
                brick_type = instance.brick_type
                all_scene_snaps |= set(
                        (instance_id, i)
                        for i in range(len(brick_type.snaps)))
            #---------------------------------------------------------------
            # build a list of occupied snaps
            all_snap_connections = scene.get_all_snap_connections()
            occupied_snaps = set()
            for a_id, connections in all_snap_connections.items():
                for b_id, b_snap, a_snap in connections:
                    occupied_snaps.add((a_id, a_snap))
                    occupied_snaps.add((b_id, b_snap))
            #---------------------------------------------------------------
            # build a list of unoccupied snaps
            unoccupied_snaps = all_scene_snaps - occupied_snaps
            unoccupied_snaps = [
                    scene.instances[instance_id].get_snap(snap_id)
                    for instance_id, snap_id in unoccupied_snaps]
            unoccupied_snaps = [
                    snap for snap in unoccupied_snaps]
                    #if (isinstance(snap, SnapCylinder) and
                    #        snap.contains_stud_radius())]
            
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
                for brick_type, transform in sub_assembly:
                    color = random.choice(colors)
                    scene.add_brick_type(brick_type)
                    new_instance = scene.add_instance(
                            brick_type, color, transform)
                    new_instances.append(new_instance)
                    new_snaps = new_instance.get_snaps()
                    new_good_snaps = [
                            snap for snap in new_snaps]
                            #if (isinstance(snap, SnapCylinder) and
                            #        snap.contains_stud_radius())]
                    sub_assembly_snaps.extend(new_good_snaps)
                
                # TMP to handle bad sub-assemblies
                if len(sub_assembly_snaps) == 0:
                    for instance in new_instances:
                        scene.remove_instance(instance)
                    continue
                
                #===============================================================
                # try to find a valid connection
                #---------------------------------------------------------------
                # get all pairs
                scene_snaps = unoccupied_snaps
                #if len(scene_snaps) > 100:
                #    scene_snaps = random.sample(scene_snaps, 100)
                pairs = list(get_all_transformed_snap_pairs(
                        unoccupied_snaps, sub_assembly_snaps))
                if len(pairs) == 0:
                    for instance in new_instances:
                        scene.remove_instance(instance)
                    continue
                #---------------------------------------------------------------
                # try to find a pair that is not in collision
                for j in range(retries):
                    scene_snap, sub_assembly_snap = random.choice(pairs)
                    offset_transform = (
                            scene_snap.transform @
                            numpy.linalg.inv(sub_assembly_snap.transform))
                    
                    # make sure the offset transform is unit volume
                    # some where somehow getting scaled
                    det = numpy.linalg.det(offset_transform[:3,:3])
                    if abs(det-1) > 0.1:
                        continue
                    
                    for instance in new_instances:
                        new_transform = offset_transform @ instance.transform
                        scene.set_instance_transform(instance, new_transform)
                    if debug:
                        dump_images = '%i_%i'%(i,j)
                    else:
                        dump_images = None
                    collision = check_collision(
                            scene,
                            new_instances,
                            scene_snap.transform,
                            sub_assembly_snap.polarity,
                            frame_buffer=frame_buffer,
                            dump_images=dump_images,
                    )
                    if debug:
                        scene.export_ldraw('%i_%s.ldr'%(i,j))
                    if not collision:
                        break
                    else:
                        for instance, (_, transform) in zip(
                                new_instances, sub_assembly):
                            scene.set_instance_transform(
                                str(instance),
                                transform,
                            )
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
                for brick_type, transform in sub_assembly:
                    color = random.choice(colors)
                    scene.add_brick_type(brick_type)
                    new_instance = scene.add_instance(
                            brick_type, color, transform)
                    new_instances.append(new_instance)
                    new_snaps = new_instance.get_snaps()
                    new_good_snaps = [
                            snap for snap in new_snaps]
                            #if (isinstance(snap, SnapCylinder) and
                            #    snap.contains_stud_radius())]
                    sub_assembly_snaps.extend(new_good_snaps)
                
                if len(sub_assembly_snaps):
                    break
                else:
                    for instance in new_instances:
                        scene.remove_instance(instance)
    
    return scene
