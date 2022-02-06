import numpy

from ltron.exceptions import LtronException
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.bricks.snap import SnapCylinder
from ltron.geometry.utils import unscale_transform

class NoUprightableSnaps(LtronException):
    pass

class UprightSceneComponent(LtronGymComponent):
    def __init__(self, scene_component, fail_mode='ignore'):
        self.scene_component = scene_component
        self.fail_mode = fail_mode
    
    def reset(self):
        scene = self.scene_component.brick_scene
        for i, instance in scene.instances.items():
            if len(instance.get_upright_snaps()):
                return None
        
        for i, instance in scene.instances.items():
            snaps = instance.snaps
            for snap in snaps:
                if isinstance(snap, SnapCylinder):
                    if snap.polarity == '+':
                        new_transform = scene.upright
                    elif snap.polarity == '-':
                        new_transform = numpy.eye(4)
                    
                    snap_transform = unscale_transform(snap.transform)
                    offset = new_transform @ numpy.linalg.inv(snap_transform)
                    
                    for j, instance in scene.instances.items():
                        new_instance_transform = offset @ instance.transform
                        scene.move_instance(j, new_instance_transform)
                    
                    return
        
        if self.fail_mode == 'ignore':
            pass
        elif self.fail_mode == 'raise':
            raise NoUprightableSnaps(
                'Could not find any snaps to make the scene upright')
