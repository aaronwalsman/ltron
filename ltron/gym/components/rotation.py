import numpy
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from gym.spaces import (
    Discrete,
    Tuple,
    Dict,
    MultiDiscrete
)
from ltron.gym.spaces import (
    SinglePixelSelectionSpace,
)
import math

class RotationAroundSnap(LtronGymComponent):
    def __init__(self, sceneComp, pos_snap_render, neg_snap_render):
        self.scene_component = sceneComp
        self.pos_snap_render = pos_snap_render
        self.neg_snap_render = neg_snap_render

    def reset(self):
        return None, 0, False, None

    def transform_about_snap(self, instance_id, snap_id, transform, scene):
        instance = scene.instances[instance_id]
        snap_transform = instance.get_snap(snap_id).transform
        prototype_transform = instance.brick_type.snaps[snap_id].transform
        instance_transform = (
                snap_transform @
                transform @
                numpy.linalg.inv(prototype_transform))
        scene.move_instance(instance, instance_transform)

    def step(self, action):

        polarity, x_cord, y_cord, degree = action[0], action[1], action[2], action[3]
        trans = numpy.eye(4)
        rotate_x = trans.clone()
        rotate_x[1,1] = math.cos(degree)
        rotate_x[1,2] = -math.sin(degree)
        rotate_x[2:1] = math.sin(degree)
        rotate_x[2:2] = math.cos(degree)

        rotate_y = trans.clone()
        rotate_y[0,0] = math.cos(degree)
        rotate_y[0,2] = math.sin(degree)
        rotate_y[2,0] = -math.sin(degree)
        rotate_y[2,2] = math.cos(degree)

        rotate_z = trans.clonse()
        rotate_z[0,0] = math.cos(action)
        rotate_z[0,1] = -math.sin(action)
        rotate_z[1,0] = math.sin(action)
        rotate_z[1,1] = math.cos(action)

        if polarity == 1:
            rotate_map = self.pos_snap_render.observation
        else:
            rotate_map = self.neg_snap_render.observation

        instance, snap_id = rotate_map[x_cord, y_cord]
        self.transform_about_snap(instance, snap_id, rotate_y, self.scene_component.brick_scene)

        return None, 0, False, None