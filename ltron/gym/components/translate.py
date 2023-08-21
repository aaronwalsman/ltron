import numpy

from ltron.gym.components import TransformSnapComponent

class OrthogonalCameraSpaceTranslateComponent(TransformSnapComponent):
    def __init__(self,
        scene_component,
        check_collision,
        translates=(
            (  0,  0,  0),
            ( -8,  0,  0),
            (  8,  0,  0),
            (-20,  0,  0),
            ( 20,  0,  0),
            (-24,  0,  0),
            ( 24,  0,  0),
            (-48,  0,  0),
            ( 48,  0,  0),
            (-80,  0,  0),
            ( 80,  0,  0),
            (  0, -8,  0),
            (  0,  8,  0),
            (  0,-20,  0),
            (  0, 20,  0),
            (  0,-24,  0),
            (  0, 24,  0),
            (  0,-48,  0),
            (  0, 48,  0),
            (  0,-80,  0),
            (  0, 80,  0),
            (  0,  0, -8),
            (  0,  0,  8),
            (  0,  0,-20),
            (  0,  0, 20),
            (  0,  0,-24),
            (  0,  0, 24),
            (  0,  0,-48),
            (  0,  0, 48),
            (  0,  0,-80),
            (  0,  0, 80),
        ),
    ):
        transforms = []
        for translate in translates:
            transform = numpy.eye(4)
            transform[:3,3] = translate
            transforms.append(transform)
        super().__init__(
            scene_component,
            check_collision,
            transforms,
            space='projected_camera',
        )
    
    def no_op_action(self):
        return 0

class CursorOrthogonalCameraSpaceTranslateComponent(
    OrthogonalCameraSpaceTranslateComponent
):
    def __init__(self, scene_component, cursor_component, check_collision):
        super().__init__(scene_component, check_collision)
        self.cursor_component = cursor_component
    
    def step(self, action):
        if not action:
            return None, 0, False, False, {}
        
        instance_id, snap_id = self.cursor_component.click_snap
        avoid_collision = super().transform_snap(instance_id, snap_id, action)
        
        return None, 0., False, False, {}
