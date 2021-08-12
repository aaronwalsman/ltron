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

from ltron.geometry.collision import check_collision

class PickandPlace(LtronGymComponent):
    def __init__(self, scene, pos_snap_render, neg_snap_render):
        self.scene_component = scene
        self.action_executed = 0
        self.pos_snap_render = pos_snap_render
        self.neg_snap_render = neg_snap_render
        self.instance_pos = {}
        self.width = self.pos_snap_render.width
        self.height = self.pos_snap_render.height
        assert self.neg_snap_render.width == self.width
        assert self.neg_snap_render.height == self.height
        
        activate_space = Discrete(2)
        pick_polarity_space = Discrete(2)
        pick_space = SinglePixelSelectionSpace(self.width, self.height)
        place_space = SinglePixelSelectionSpace(self.width, self.height)
        self.action_space = Tuple(
            (activate_space, pick_polarity_space, pick_space, place_space))
        #self.action_space = MultiDiscrete([2, self.width, self.height, self.width, self.height])
        # self.action_space = MultiDiscrete([2, 20, 20, 20, 20])

        self.observation_space = Dict({'success': Discrete(2)})

    def reset(self):
        return {'success':False}

    def step(self, action):
        #if action is None: return None, 0, False, None
        #polarity = action[0]  # Integer, 0 or 1
        # pick_x, pick_y = action[1]+100, action[2]+100 # a tuple, corresponding to the coordinate of pick location
        # place_x, place_y = action[3]+150, action[4]+150 # a tuple, corresponding to the coordinate of place location
        #pick_x, pick_y = action[1], action[2]
        #place_x, place_y = action[3], action[4]
        
        activate, polarity, (pick_y, pick_x), (place_y, place_x) = action
        
        if polarity == 1:
            pick_map = self.pos_snap_render.observation
            place_map = self.neg_snap_render.observation
        else:
            pick_map = self.neg_snap_render.observation
            place_map = self.pos_snap_render.observation
        
        pick_instance, pick_id = pick_map[pick_y, pick_x]
        place_instance, place_id = place_map[place_y, place_x]
        # if check_collision(self.scene_component.brick_scene, pick_instance, abs(polarity - 1), (self.width, self.height)):
        #     return {'pick_place_succeed': 0}, 0, False, None
        
        if pick_instance == 0 or place_instance == 0:
            return {'success' : 0}, 0, False, None
        
        self.scene_component.brick_scene.pick_and_place_snap(
            (pick_instance, pick_id), (place_instance, place_id))
        
        return {'success' : 1}, 0, False, None  # the observation is whether the action succeeds or not
