import math

from pyquaternion import Quaternion

#from ltron.geometry.collision import check_collision
from ltron.geometry.utils import orthogonal_orientations
from ltron.gym.components import TransformSnapComponent

class OrthogonalCameraSpaceRotationComponent(TransformSnapComponent):
    def __init__(self, scene_component, check_collision):
        transforms = orthogonal_orientations()
        super().__init__(
            scene_component,
            check_collision,
            transforms,
            space='projected_camera',
        )
    
    def no_op_action(self):
        return 0

class RotateSnapAboutAxisComponent(TransformSnapComponent):
    def __init__(
        self,
        scene_component,
        check_collision,
        rotate_steps=4,
        rotate_axis=(0,1,0),
    ):
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
        super().__init__(
            scene_component, check_collision, transforms, space='local')
    
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
        check_collision=True,
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
