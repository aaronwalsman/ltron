import math

import numpy

from pyquaternion import Quaternion

from gym.spaces import (
    Dict,
    Tuple,
    Discrete,
    Box,
)

from ltron.gym.spaces import (
    SingleSnapIndexSpace,
    SinglePixelSelectionSpace,
)
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class PickAndPlaceSymbolic(LtronGymComponent):
    def __init__(self, scene_component, max_instances, max_snaps):
        self.scene_component = scene_component
        self.max_instances = max_instances
        self.max_snaps = max_snaps
        
        activate_space = Discrete(2)
        pick_space = SingleSnapIndexSpace(max_instances, max_snaps)
        place_space = SingleSnapIndexSpace(max_instances, max_snaps)
        self.action_space = Dict({
            'pick':pick_space,
            'place':place_space,
        })
    
    def step(self, action):
        pick = action['pick']
        place = action['place']
        if pick[0] and place[0]:
            self.scene_component.brick_scene.pick_and_place_snap(pick, place)
        
        return None, 0., False, {}

class VectorOffsetSymbolic(LtronGymComponent):
    def __init__(self,
        scene_component,
        max_instances,
        max_snaps,
        step_x=20,
        step_y=8,
        step_z=20,
        step_theta=math.pi/2.,
        space='world',
        snap_to_axis=True,
    ):
        self.scene_component = scene_component
        self.max_instances = max_instances
        self.max_snaps = max_snaps
        self.step_x = step_x
        self.step_y = step_y
        self.step_z = step_z
        self.step_theta = step_theta
        self.space = space
        self.snap_to_axis = snap_to_axis
        
        snap_space = SingleSnapIndexSpace(max_instances, max_snaps)
        direction_space = Box(-1,1,shape=(3,))
        motion_space = Discrete(2)
        self.action_space = Dict({
            'pick':snap_space,
            'direction':direction_space,
            'motion':motion_space
        })
        
    def step(self, action):
        instance_id, snap_id = action['pick']
        if instance_id == 0:
            return None, 0., False, {}
        direction = action['direction']
        motion = action['motion']
        direction /= numpy.linalg.norm(direction)
        direction = numpy.concatenate((direction, [0]))
        
        scene = self.scene_component.brick_scene
        instance = scene.instances[instance_id]
        snap = instance.snaps[snap_id]
        
        if self.space == 'world':
            pass
        elif self.space == 'local':
            direction = snap.transform @ direction
        elif self.space == 'camera':
            view_matrix = scene.get_view_matrix()
            camera_pose = numpy.linalg.inv(view_matrix)
            direction = camera_pose @ direction
        
        if self.snap_to_axis:
            direction = numpy.linalg.inv(snap.transform) @ direction
            
            closest_axis = numpy.argmax(numpy.abs(direction))
            new_direction = [0,0,0,0]
            if direction[closest_axis] < 0:
                new_direction[closest_axis] = -1
            else:
                new_direction[closest_axis] = 1
            direction = snap.transform @ new_direction
        
        if motion == 0:
            # translate
            instance_transform = instance.transform
            instance_transform[0,3] += direction[0] * self.step_x
            instance_transform[1,3] += direction[1] * self.step_y
            instance_transform[2,3] += direction[2] * self.step_z
            scene.move_instance(instance, instance_transform)
        
        elif motion == 1:
            # rotate
            rotation = Quaternion(axis=direction[:3], angle=self.step_theta)
            snap_position = snap.transform[0:3,3]
            instance_transform = instance.transform
            instance_transform[0:3,3] -= snap_position
            instance_transform = (
                rotation.transformation_matrix @ instance_transform)
            instance_transform[0:3,3] += snap_position
            scene.move_instance(instance, instance_transform)
        
        return None, 0., False, {}

class LocalOffsetSymbolic(LtronGymComponent):
    def __init__(self,
        scene_component,
        max_instances,
        max_snaps,
        step_x=20,
        step_y=8,
        step_z=20,
        step_theta=math.pi/2.,
    ):
        self.scene_component = scene_component
        self.max_instances = max_instances
        self.max_snaps = max_snaps
        self.step_x = step_x
        self.step_y = step_y
        self.step_z = step_z
        self.step_theta = step_theta
        
        snap_space = SingleSnapIndexSpace(max_instances, max_snaps)
        motion_space = Discrete(9)
        self.action_space = Dict({
            'snap':snap_space,
            'motion':motion_space,
        })
    
    def step(self, action):
        instance_id, snap_id = action['snap']
        motion = action['motion']
        if motion == 0:
            pass
        elif motion == 1:
            # -x
            pass
        
        elif motion == 2:
            # +x
            pass
        
        elif motion == 3:
            # -y
            pass
        
        elif motion == 4:
            # +y
            pass
        
        elif motion == 5:
            # -z
            pass
        
        elif motion == 6:
            # +z
            pass
        
        elif motion == 7:
            # -theta
            pass
        
        elif motion == 8:
            # +theta
            pass
        
        return None, 0., False, {}
