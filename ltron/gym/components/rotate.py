import math

import numpy

from pyquaternion import Quaternion

from gymnasium.spaces import (
    Discrete,
    Tuple,
    Dict,
    MultiDiscrete,
)

from supermecha import SuperMechaComponent
from ltron.geometry.collision import check_collision
from ltron.geometry.utils import orthogonal_orientations

class TransformSnapComponent(SuperMechaComponent):
    def __init__(
        self,
        scene_component,
        check_collision,
        transforms,
        space='local',
    ):
        self.scene_component = scene_component
        self.check_collision = check_collision
        self.transforms = transforms
        self.space = space
        self.action_space = Discrete(len(transforms))
    
    def transform_snap(self, instance_id, snap_id, action):
        if instance_id == 0:
            return False
        
        transform = self.transforms[action]
        
        scene = self.scene_component.brick_scene
        instance = scene.instances[instance_id]
        if snap_id >= len(instance.snaps):
            return False
        snap = instance.snaps[snap_id]
        
        scene.transform_about_snap(
            [instance],
            snap,
            transform,
            check_collision=self.check_collision,
            space=self.space,
        )
        
        return True

class OrthogonalCameraSpaceRotationComponent(TransformSnapComponent):
    def __init__(self, scene_component, check_collisions):
        transforms = orthogonal_orientations()
        super().__init__(
            scene_component,
            check_collision,
            transforms,
            space='projected_camera',
        )
    
    def no_op_action(self):
        return 0

class ReducedOrthogonalCameraSpaceRotationComponent(TransformSnapComponent):
    def __init__(self, scene_component, check_collisions):
        transforms = []
        super().__init__(
            scene_component,
            check_collisions,
            transforms,
            space='projected_camera',
        )

class RotateSnapAboutAxisComponent(TransformSnapComponent):
    def __init__(
        self,
        scene_component,
        check_collision,
        rotate_steps=4,
        rotate_axis=(0,1,0),
    ):
        #self.scene_component = scene_component
        #self.check_collision = check_collision
        rotate_step_size = math.pi * 2 / rotate_steps
        self.rotate_step_size = rotate_step_size
        self.rotate_axis = rotate_axis
        transforms = [
            Quaternion(
                axis=self.rotate_axis,
                angle=i*rotate_step_size
            ).transformation_matrix
            for i in range(rotate_steps)
        ]
        #self.action_space = Discrete(rotate_steps)
        super().__init__(
            scene_component, check_collision, transforms, space='local')
    
    '''
    def rotate_snap(self, instance_id, snap_id, action):
        success = False
        if instance_id == 0:
            return success
        
        #if action == 2:
        #    action = -1
        angle = self.rotate_step_size*action
        rotation = Quaternion(
            axis=self.rotate_axis, angle=angle).transformation_matrix
        
        scene = self.scene_component.brick_scene
        instance = scene.instances[instance_id]
        if snap_id >= len(instance.snaps):
            return False
        snap = instance.snaps[snap_id]
        
        scene.transform_about_snap(
            [instance], snap, rotation, check_collision=self.check_collision)
    '''
    def no_op_action(self):
        return 0

class CursorOrthogonalCameraSpaceRotationComponent(
    OrthogonalCameraSpaceRotationComponent
):
    def __init__(self, scene_component, cursor_component, check_collision):
        super().__init__(scene_component, check_collision)
        self.cursor_component = cursor_component
    
    def step(self, action):
        if not action:
            return None, 0, False, False, {}
        
        instance_id, snap_id = self.cursor_component.click_snap
        super().transform_snap(instance_id, snap_id, action)
        
        return None, 0., False, False, {}

class CursorRotateSnapAboutAxisComponent(RotateSnapAboutAxisComponent):
    def __init__(self,
        scene_component,
        cursor_component,
        #overlay_component,
        check_collision=True,
        #rotate_step_size=math.radians(90.),
        rotate_steps=4,
        rotate_axis=(0,1,0),
    ):
        super().__init__(
            scene_component,
            #overlay_component,
            check_collision=check_collision,
            rotate_steps=rotate_steps,
            rotate_axis=rotate_axis,
        )
        self.cursor_component = cursor_component
    
    def step(self, action):
        
        if not action:
            return None, 0, False, False, {}
        
        instance_id, snap_id = self.cursor_component.click_snap
        
        super().transform_snap(instance_id, snap_id, action)
        
        return None, 0., False, False, {}
    
    def no_op_action(self):
        return 0



'''
class RotateAboutSnap(LtronGymComponent):
    def __init__(
        self,
        scene_components,
        cursor_component,
        check_collision,
        rotation_steps = 4,
        allow_snap_flip = False,
    ):
        self.scene_components = scene_components
        self.cursor_component = cursor_component
        self.check_collision = check_collision
        self.rotation_steps = rotation_steps
        self.allow_snap_flip = allow_snap_flip
        #self.action_space = Dict({
            #'activate':Discrete(2),
        #    'rotation':Discrete(self.rotation_steps),
        #})
        if allow_snap_flip:
            self.action_space = Discrete(self.rotation_steps*2)
        else:
            self.action_space = Discrete(self.rotation_steps)

        #self.observation_space = Dict({'success': Discrete(2)})
    
    #def reset(self):
    #    return {'success':0}
    
    def step(self, action):
        #failure = {'success' : 0}, 0, False, None
        if not action:
            #return failure
            return None, 0., False, {}
        
        a = action % self.rotation_steps
        degree = a * math.pi * 2 / self.rotation_steps
        flip = action // self.rotation_steps
        
        trans = numpy.eye(4)
        
        rotate_y = numpy.copy(trans)
        rotate_y[0,0] = math.cos(degree)
        rotate_y[0,2] = math.sin(degree)
        rotate_y[2,0] = -math.sin(degree)
        rotate_y[2,2] = math.cos(degree)
        
        if flip:
            flip_transform = numpy.array([
                [-1, 0, 0, 0],
                [ 0,-1, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1]
            ])
            rotate_y = rotate_y @ flip_transform
        
        n, i, s = self.cursor_component.get_selected_snap()
        if i == 0:
            #return failure
            return None, 0., False, {}
        
        scene_component = self.scene_components[n]
        scene = scene_component.brick_scene
        instance = scene.instances[i]
        if s >= len(instance.snaps):
            #return failure
            return None, 0., False, {}
        original_instance_transform = instance.transform
        snap = instance.snaps[s]
        scene.transform_about_snap([instance], snap, rotate_y)
        
        if self.check_collision:
            transform = instance.transform
            snap = instance.snaps[s]
            collision = scene_component.brick_scene.check_snap_collision(
                target_instances=[instance], snap=snap)
            if collision:
                scene_component.brick_scene.move_instance(
                    instance, original_instance_transform)
                #return {'success' : 0}, 0, False, None
                return None, 0., False, {}
        
        #return {'success' : 1}, 0, False, None
        return None, 0., False, {}
    
    def no_op_action(self):
        return 0
'''
