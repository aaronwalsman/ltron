import math
from collections import OrderedDict

import numpy

import gym
import gym.spaces as spaces
import numpy

from ltron.config import Config
from ltron.dataset.paths import get_dataset_info
from ltron.gym.envs.ltron_env import LtronEnv
from ltron.gym.components.scene import EmptySceneComponent
from ltron.gym.components.episode import EpisodeLengthComponent
from ltron.gym.components.loader import DatasetLoaderComponent
from ltron.gym.components.cursor_action_wrapper import (
    CursorActionWrapperConfig, CursorActionWrapper)
from ltron.gym.components.render import ColorRenderComponent
from ltron.gym.components.spot_check import SpotCheck, ConstantImage
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.edit_distance import EditDistance
from ltron.gym.components.assembly import AssemblyComponent
from ltron.gym.components.upright import UprightSceneComponent
from ltron.gym.components.tile import DeduplicateTileMaskComponent
from ltron.gym.components.partial_disassembly import (
    MultiScreenPartialDisassemblyComponent,
)
from ltron.gym.components.build_expert import BuildExpert

class MultiScreenEditEnvConfig(CursorActionWrapperConfig):
    dataset = 'random_construction_6_6'
    split = 'train'
    subset = None
    
    table_image_height = 256
    table_image_width = 256
    hand_image_height = 96
    hand_image_width = 96
    
    dataset_sample_mode = 'uniform'
    
    max_episode_length = 32
    
    randomize_colors = True
    
    include_score = True
    
    check_collision = True
    train = True
    
    tile_color_render = True
    tile_width = 16
    tile_height = 16
    
    egl_device=None

class MultiScreenEditEnv(LtronEnv):
    def __init__(self, config, rank=0, size=1, print_traceback=False):
        components = OrderedDict()
        
        # Setup Components =====================================================
        # dataset info
        dataset_info = get_dataset_info(config.dataset)
        shape_ids = dataset_info['shape_ids']
        color_ids = dataset_info['color_ids']
        max_instances = dataset_info['max_instances_per_scene']
        max_edges = dataset_info['max_edges_per_scene']
        
        # scenes
        components['table_scene'] = EmptySceneComponent(
            shape_ids,
            color_ids,
            max_instances,
            max_edges,
            track_snaps=True,
            collision_checker=config.check_collision,
        )
        components['hand_scene'] = EmptySceneComponent(
            shape_ids,
            color_ids,
            max_instances,
            max_edges,
            track_snaps=True,
            collision_checker=config.check_collision,
        )
        
        # dataset
        components['dataset'] = DatasetLoaderComponent(
            components['table_scene'],
            config.dataset,
            config.split,
            subset=config.subset,
            rank=rank,
            size=size,
            sample_mode=config.dataset_sample_mode,
        )
        
        # uprightify
        components['upright'] = UprightSceneComponent(
            scene_component = components['table_scene'])
        
        # time step
        components['step'] = EpisodeLengthComponent(
            config.max_episode_length, observe_step=True)
        
        # color randomization
        if config.randomize_colors:
            components['color_randomization'] = RandomizeColorsComponent(
                dataset_info['color_ids'],
                components['table_scene'],
                randomize_frequency='reset',
            )
        
        # initial assembly
        components['initial_table_assembly'] = AssemblyComponent(
            components['table_scene'],
            shape_ids,
            color_ids,
            max_instances,
            max_edges,
            update_frequency='reset',
            observe_assembly=config.train,
        )
        
        # Action ===============================================================
        components['action'] = CursorActionWrapper(
            config,
            components['table_scene'],
            components['hand_scene'],
            max_instances,
            print_traceback=print_traceback,
        )
        
        # No action space, but needs renderers =================================
        
        # initial color render component
        components['initial_table_color_render'] = ColorRenderComponent(
            config.table_image_width,
            config.table_image_height,
            components['table_scene'],
            render_frequency='reset',
        )
        '''
        components['initial_table_color_render'] = ConstantImage(
            config.table_image_width, config.table_image_height)
        '''
        components['initial_table'] = DeduplicateTileMaskComponent(
            config.tile_width,
            config.tile_height,
            components['initial_table_color_render'],
        )
        
        # partial disassembly
        partial_disassembly_component = MultiScreenPartialDisassemblyComponent(
            #action_components['pick_and_place'],
            components['action'].components['pick_and_place'],
            components['action'].components['table_pos_snap_render'],
            components['action'].components['table_neg_snap_render'],
            ['table'],
            ['hand'],
            max_instances,
            num_disassembly_steps=1,
        )
        components['partial_disassembly'] = partial_disassembly_component
        
        # Always-on renderers/observation spaces ===============================
        # color render
        components['table_color_render'] = ColorRenderComponent(
            config.table_image_width,
            config.table_image_height,
            components['table_scene'],
            anti_alias=True,
            observable=False,
        )
        '''
        components['table_color_render'] = SpotCheck(
            config.table_image_width,
            config.table_image_height,
        )
        '''
        components['hand_color_render'] = ColorRenderComponent(
            config.hand_image_width,
            config.hand_image_height,
            components['hand_scene'],
            anti_alias=True,
            observable=False,
        )
        '''
        components['hand_color_render'] = ConstantImage(
            config.hand_image_width, config.hand_image_height)
        '''
        components['table'] = DeduplicateTileMaskComponent(
            config.tile_width,
            config.tile_height,
            components['table_color_render'],
        )
        components['hand'] = DeduplicateTileMaskComponent(
            config.tile_width,
            config.tile_height,
            components['hand_color_render'],
        )
        
        # current assembly
        components['table_assembly'] = AssemblyComponent(
            components['table_scene'],
            shape_ids,
            color_ids,
            max_instances,
            max_edges,
            update_frequency = 'step',
            observe_assembly = config.train,
        )
        
        components['hand_assembly'] = AssemblyComponent(
            components['hand_scene'],
            shape_ids,
            color_ids,
            max_instances,
            max_edges,
            update_frequency = 'step',
            observe_assembly = config.train,
        )
        
        # score
        components['score'] = EditDistance(
            components['initial_table_assembly'],
            components['table_assembly'],
            shape_ids,
        )
        
        shape_names = {v:k for k,v in shape_ids.items()}
        if config.train:
            components['expert'] = BuildExpert(
                components['action'],
                {'table':components['table_scene'],
                 'hand':components['hand_scene']},
                components['initial_table_assembly'],
                {'table':components['table_assembly'],
                 'hand':components['hand_assembly']},
                'table',
                shape_names,
            )
        
        super().__init__(
            components,
            combine_action_space='single',
            print_traceback=print_traceback,
        )
