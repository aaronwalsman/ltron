import math
from collections import OrderedDict

import gym
import gym.spaces as spaces

from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.render import (
        ColorRenderComponent, SegmentationRenderComponent, SnapRenderComponent)
from ltron.gym.components.disassembly import PixelDisassemblyComponent
from ltron.gym.components.rotation import RotationAroundSnap
from ltron.gym.components.pick_and_place import PickandPlace
from ltron.gym.components.viewpoint import (
        ControlledAzimuthalViewpointComponent)
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.reassembly_score import ReassemblyScore

class SimplifiedReassemblyWrapper(gym.Env):
    def __init__(*args, **kwargs):
        self.env = reassembly_env(*args, **kwargs)
        
        # setup action space
        render_component = self.env.components['color_render']
        height = render_component.height
        width = render_component.width
        num_modes = (
            6 + # camera motion
            1 + # disassembly
            1 + # rotate
            1 + # pick and place
            1 + # start disassembly
        )
        self.action_space = spaces.MultiDiscrete(
            num_modes, height, width, height, width)
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=numpy.uint8)
    
    def reset(self):
        observation = self.env.reset()
        observation = self.convert_observation(observation)
        return observation
    
    def step(self, action):
        action = self.convert_action(action)
        observation, reward, terminal, info = self.env.step(action)
        observation = self.convert_observation(observation)
        return observation, reward, terminal, info
    
    def convert_observation(self, observation):
        return observation['color_render']
    
    def convert_action(self, action):
        mode, pick_y, pick_x, place_y, place_x = action
        dict_action = {}
        
        # viewpoint
        if mode < 6:
            viewpoint_action = mode + 1
        else:
            viewpoint_action = 0
        dict_action['viewpoint'] = viewpoint_action
        
        # 
        

def reassembly_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
    image_width=256,
    image_height=256,
    map_width=64,
    map_height=64,
    dataset_reset_mode='uniform',
    randomize_colors=True,
    check_collisions=True,
    print_traceback=True,
):
    components = OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
        dataset, split, subset, rank, size, reset_mode=dataset_reset_mode)
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    
    # scene
    components['scene'] = SceneComponent(
        dataset_component=components['dataset'],
        path_location=[0],
        track_snaps=True,
        collision_checker=check_collisions,
    )
    
    # viewpoint
    components['viewpoint'] = ControlledAzimuthalViewpointComponent(
        components['scene'],
        azimuth_steps=8,
        elevation_range=[math.radians(-30), math.radians(30)],
        elevation_steps=2,
        distance_range=[200, 200],
        distance_steps=1,
        aspect_ratio=image_width/image_height,
    )
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponent(
            dataset_info['all_colors'],
            components['scene'],
            randomize_frequency='scene',
        )
    
    # utility rendering components
    pos_snap_render = SnapRenderComponent(
        map_width, map_height, components['scene'], polarity='+')
    neg_snap_render = SnapRenderComponent(
        map_width, map_height, components['scene'], polarity='-')
    
    # action spaces
    components['remove'] = PixelDisassemblyComponent(
        components['scene'],
        pos_snap_render,
        neg_snap_render,
        check_collisions=check_collisions,
    )
    components['rotation'] = RotationAroundSnap(
        components['scene'],
        pos_snap_render,
        neg_snap_render,
        check_collisions=check_collisions,
    )
    components['pick_and_place'] = PickandPlace(
        components['scene'],
        pos_snap_render,
        neg_snap_render,
        check_collisions=check_collisions,
    )
    
    # reassembly
    components['reconstruction_score'] = Reassembly(
        components['scene'])
    
    # color render
    components['color_render'] = ColorRenderComponent(
        image_width, image_height, components['scene'], anti_alias=True)
    
    # snap render
    components['pos_snap_render'] = pos_snap_render
    components['neg_snap_render'] = neg_snap_render
    
    # build the env
    env = LtronEnv(components, print_traceback=print_traceback)
    
    return env
