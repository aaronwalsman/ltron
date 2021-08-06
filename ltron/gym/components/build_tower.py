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

class TallestTower(LtronGymComponent):
    def __init__(self, scenecomponent):
        self.scenecomponent = scenecomponent

    def reset(self):
        return None

    def compute_reward(self):

        instance_tran = {}
        for k, v in self.scenecomponent.brick_scene.instances.instances.items():
            instance_tran[k] = v.transform

        instance_pos = {}
        for k, v in self.scenecomponent.brick_scene.instances.instances.items():
            instance_pos[k] = v.brick_type.bbox

        point = []
        for ins, bbox in instance_pos.items():
            minb = bbox[0]
            maxb = bbox[1]
            point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], minb[1], minb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], minb[1], minb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], maxb[1], minb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], minb[1], maxb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], maxb[1], minb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([minb[0], maxb[1], maxb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], minb[1], maxb[2], 1])))[:3])
            point.append((numpy.matmul(instance_tran[ins], numpy.array([maxb[0], maxb[1], maxb[2], 1])))[:3])

        min_y = 100000
        max_y = -1000000
        for p in point:
            if p[1] > max_y:
                max_y = p[1]
            if p[1] < min_y:
                min_y = p[1]

        # if abs(max_y - min_y) - 35 > 0: return 10000
        # else: return -1000
        return abs(max_y - min_y)

    def step(self, action):
        return None, self.compute_reward(), False, None

class PickandPlace(LtronGymComponent):
    def __init__(self, scene, pos_snap_render, neg_snap_render):
        self.scene_component = scene
        self.action_executed = 0
        self.pos_snap_render = pos_snap_render
        self.neg_snap_render = neg_snap_render
        self.instance_pos = {}
        width = self.pos_snap_render.width
        height = self.pos_snap_render.height
        assert self.neg_snap_render.width == width
        assert self.neg_snap_render.height == height

        pick_polarity_space = Discrete(2)
        pick_space = SinglePixelSelectionSpace(width, height)
        place_space = SinglePixelSelectionSpace(width, height)
        # self.action_space = Tuple(
        #     (pick_polarity_space, pick_space, place_space))
        self.action_space = MultiDiscrete([2, width, height, width, height])
        # self.action_space = MultiDiscrete([2, 20, 20, 20, 20])

        self.observation_space = Dict({'pick_place_succeed': Discrete(2)})

    def reset(self):
        return None, 0, False, None

    def step(self, action):
        if action is None: return None, 0, False, None
        polarity = action[0]  # Integer, 0 or 1
        # pick_x, pick_y = action[1]+100, action[2]+100 # a tuple, corresponding to the coordinate of pick location
        # place_x, place_y = action[3]+150, action[4]+150 # a tuple, corresponding to the coordinate of place location
        pick_x, pick_y = action[1], action[2]
        place_x, place_y = action[3], action[4]

        if polarity == 1:
            pick_map = self.pos_snap_render.observation
            place_map = self.neg_snap_render.observation
        else:
            pick_map = self.neg_snap_render.observation
            place_map = self.pos_snap_render.observation

        pick_instance, pick_id = pick_map[pick_x, pick_y]
        place_instance, place_id = place_map[place_x, place_y]

        if pick_instance == 0 or pick_id == 0 or place_instance == 0 or place_id == 0:
            return {'pick_place_succeed' : 0}, 0, False, None

        self.scene_component.brick_scene.pick_and_place_snap((pick_instance, pick_id), (place_instance, place_id))

        return {'pick_place_suceed' : 1}, 0, False, None  # the observation is whether the action succeeds or not

class RotationAroundSnap(LtronGymComponent):
    def __init__(self, scene):
        self.scene_component = scene

    def reset(self):
        return None, 0, False, None

    def step(self, action):
        x_cord, y_cord, degree = action[0], action[1], action[2]
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

        prev = self.scene_component.brick_scene.get_all_snap_connections

        return None, 0, False, None