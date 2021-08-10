import numpy
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from gym.spaces import (
    Discrete,
    Tuple,
    Dict,
    MultiDiscrete
)
from ltron.geometry.collision import check_collision
import math

class RotationAroundSnap(LtronGymComponent):
    def __init__(self, sceneComp, pos_snap_render, neg_snap_render):
        self.scene_component = sceneComp
        self.pos_snap_render = pos_snap_render
        self.neg_snap_render = neg_snap_render
        width = self.pos_snap_render.width
        height = self.pos_snap_render.height
        assert self.neg_snap_render.width == width
        assert self.neg_snap_render.height == height
        self.action_space = MultiDiscrete([2, width, height, 180])

        self.observation_space = Dict({'rotation_succeed': Discrete(2)})

    def reset(self):
        return None, 0, False, None

    def transform_about_snap(self, polarity, instance_id, snap_id, transform, scene):
        instance = scene.instances[instance_id]
        snap_transform = instance.get_snap(snap_id).transform
        prototype_transform = instance.brick_type.snaps[snap_id].transform
        instance_transform = (
                snap_transform @
                transform @
                numpy.linalg.inv(prototype_transform))

        table = scene.instances.instances
        c_polarity = '+-'[polarity]
        print('pre:', check_collision(scene, [table[instance_id]], snap_transform, c_polarity))

        scene.move_instance(instance, instance_transform)
        snap_transform = instance.get_snap(snap_id).transform
        print('post:', check_collision(scene, [instance], snap_transform, c_polarity))

    def step(self, action):

        if action is None: return {'rotation_suceed' : 0}, 0, False, None

        polarity, x_cord, y_cord, degree = action[0], action[1], action[2], action[3]
        trans = numpy.eye(4)
        rotate_x = numpy.copy(trans)
        rotate_x[1,1] = math.cos(degree)
        rotate_x[1,2] = -math.sin(degree)
        rotate_x[2:1] = math.sin(degree)
        rotate_x[2:2] = math.cos(degree)

        rotate_y = numpy.copy(trans)
        rotate_y[0,0] = math.cos(degree)
        rotate_y[0,2] = math.sin(degree)
        rotate_y[2,0] = -math.sin(degree)
        rotate_y[2,2] = math.cos(degree)

        rotate_z = numpy.copy(trans)
        rotate_z[0,0] = math.cos(degree)
        rotate_z[0,1] = -math.sin(degree)
        rotate_z[1,0] = math.sin(degree)
        rotate_z[1,1] = math.cos(degree)

        if polarity == 1:
            rotate_map = self.pos_snap_render.observation
        else:
            rotate_map = self.neg_snap_render.observation

        instance, snap_id = rotate_map[x_cord, y_cord]
        if instance == 0 or snap_id == 0:
            return {'pick_place_succeed' : 0}, 0, False, None
        self.transform_about_snap(polarity, instance, snap_id, rotate_y, self.scene_component.brick_scene)

        return {'rotation_suceed' : 1}, 0, False, None
