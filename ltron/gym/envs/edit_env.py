import math
from collections import OrderedDict

import numpy

import gym
import gym.spaces as spaces
import numpy

from ltron.config import Config
from ltron.gym.envs.ltron_env import LtronEnv
from ltron.gym.components.scene import (
    EmptySceneComponent, DatasetSceneComponent, SingleSceneComponent)
from ltron.gym.components.episode import EpisodeLengthComponent
from ltron.gym.components.loader import DatasetLoaderComponent
from ltron.gym.components.render import (
        ColorRenderComponent, InstanceRenderComponent, SnapRenderComponent)
from ltron.gym.components.cursor import MultiScreenCursor
from ltron.gym.components.disassembly import CursorDisassemblyComponent
from ltron.gym.components.rotation import MultiCursorRotateAboutSnap
from ltron.gym.components.pick_and_place import MultiScreenPickAndPlace
from ltron.gym.components.brick_inserter import HandspaceBrickInserter
from ltron.gym.components.viewpoint import (
        ControlledAzimuthalViewpointComponent)
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.break_and_make import (
        BreakAndMakePhaseSwitch,
        BreakOnlyPhaseSwitch,
        BreakAndMakeScore,
        BreakOnlyScore,
)
from ltron.gym.components.assembly import AssemblyComponent
from ltron.gym.components.upright import UprightSceneComponent
from ltron.gym.components.tile import DeduplicateTileMaskComponent
from ltron.gym.components.partial_disassembly import (
    MultiScreenPartialDisassemblyComponent,
)

class EditEnvConfig(Config):
    dataset = 'random_construction_6_6'
    split = 'train'
    subset = None
    
    table_image_height = 256
    table_image_width = 256
    hand_image_height = 96
    hand_image_width = 96
    
    table_map_height = 64
    table_map_width = 64
    hand_map_height = 24
    hand_map_width = 24
    
    dataset_selection_mode = 'uniform'
    
    max_episode_length = 32
    
    randomize_viewpoint = True
    randomize_colors = True
    
    include_score = True
    
    check_collision = True
    train = True
    
    tile_color_render = True
    tile_width = 16
    tile_height = 16
    
    egl_device=None
    
    table_distance = 320
    hand_distance = 180

class EditEnv(LtronEnv):
    def __init__(self, config, rank=0, size=1, print_traceback=False):
        components = OrderedDict()
        
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
            collision_checker=False,
        )
        
        # dataset
        components['dataset'] = DatasetLoaderComponent(
            components['table_scene'],
            config.dataset,
            config.split,
            subset=config.subset,
            rank=rank,
            size=size,
            sample_mode=config.dataset_reset_mode,
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
        
        # viewpoint
        azimuth_steps = 8
        elevation_range = [math.radians(-30), math.radians(30)]
        elevation_steps = 2
        # TODO: make this correct
        table_distance_steps = 1
        table_distance_range=[config.table_distance, config.table_distance]
        hand_distance_steps = 1
        hand_distance_range=[config.hand_distance, config.hand_distance]
        if config.randomize_viewpoint:
            start_position='uniform'
        else:
            start_position=(0,0,0)
        components['table_viewpoint'] = ControlledAzimuthalViewpointComponent(
            components['table_scene'],
            azimuth_steps=azimuth_steps,
            elevation_range=elevation_range,
            elevation_steps=elevation_steps,
            distance_range=table_distance_range,
            distance_steps=table_distance_steps,
            aspect_ratio=config.table_image_width/config.table_image_height,
            start_position=start_position,
            auto_frame='reset',
            frame_button=True,
        )
        components['hand_viewpoint'] = ControlledAzimuthalViewpointComponent(
            components['hand_scene'],
            azimuth_steps=azimuth_steps,
            elevation_range=elevation_range,
            elevation_steps=elevation_steps,
            distance_range=hand_distance_range,
            distance_steps=hand_distance_steps,
            aspect_ratio=config.hand_image_width/config.hand_image_height,
            start_position=(0,0,0),
            auto_frame='reset',
            frame_button=True
        )
        
        # utility rendering components
        table_pos_snap_render = SnapRenderComponent(
            config.table_map_width,
            config.table_map_height,
            components['table_scene'],
            polarity='+',
            render_frequency='on_demand',
        )
        table_neg_snap_render = SnapRenderComponent(
            config.table_map_width,
            config.table_map_height,
            components['table_scene'],
            polarity='-',
            render_frequency='on_demand',
        )
        hand_pos_snap_render = SnapRenderComponent(
            config.hand_map_width,
            config.hand_map_height,
            components['hand_scene'],
            polarity='+',
            render_frequency='on_demand',
        )
        hand_neg_snap_render = SnapRenderComponent(
            config.hand_map_width,
            config.hand_map_height,
            components['hand_scene'],
            polarity='-',
            render_frequency='on_demand',
        )
        
        # initial color render component
        components['initial_table_color_render'] = ColorRenderComponent(
            config.table_image_width,
            config.table_image_height,
            components['table_scene'],
            render_frequency='reset',
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
        
        # cursors
        components['pick_cursor'] = MultiScreenCursor(
            max_instances,
            [table_pos_snap_render, hand_pos_snap_render],
            [table_neg_snap_render, hand_neg_snap_render],
        )
        components['place_cursor'] = MultiScreenCursor(
            max_instances,
            [table_pos_snap_render, hand_pos_snap_render],
            [table_neg_snap_render, hand_neg_snap_render],
        )
        
        # action spaces
        components['rotate'] = MultiScreenRotateAboutSnap(
            (components['table_scene'], components['hand_scene']),
            components['pick_cursor'],
            check_collision=config.check_collision,
            allow_snap_flip=True,
        )
        components['pick_and_place'] = MultiScreenPickAndPlace(
            (components['table_scene'], components['hand_scene']),
            components['pick_cursor'],
            components['place_cursor'],
            check_collision=config.check_collision,
        )
        
        # partial disassembly
        partial_disassembly_component = MultiScreenPartialDisassemblyComponent(
            components['pick_and_place'],
            table_pos_snap_render,
            table_neg_snap_render,
            [0],
            [1],
            max_instances,
            num_disassembly_steps=1,
        )
        components['partial_disassembly'] = partial_disassembly_component
        
        # phase
        components['phase'] = BreakOnlyPhaseSwitch()
        
        # color render
        components['table_color_render'] = ColorRenderComponent(
            config.table_image_width,
            config.table_image_height,
            components['table_scene'],
            anti_alias=True,
            observable=False,
        )
        components['hand_color_render'] = ColorRenderComponent(
            config.hand_image_width,
            config.hand_image_height,
            components['hand_scene'],
            anti_alias=True,
            observable=False,
        )
        components['table_tile_mask'] = DeduplicateTileMaskComponent(
            config.tile_width,
            config.tile_height,
            components['table_color_render'],
        )
        components['hand_tile_mask'] = DeduplicateTileMaskComponent(
            config.tile_width,
            config.tile_height,
            components['hand_color_render'],
        )
        components['initial_table_tile_mask'] = DeduplicateTileMaskComponent(
            config.tile_width,
            config.tile_height,
            components['initial_table_color_render'],
        )
        
        # snap render
        components['table_pos_snap_render'] = table_pos_snap_render
        components['table_neg_snap_render'] = table_neg_snap_render
        components['hand_pos_snap_render'] = hand_pos_snap_render
        components['hand_neg_snap_render'] = hand_neg_snap_render
        
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
        components['score'] = BreakAndMakeScore(
            components['initial_table_assembly'],
            components['table_assembly'],
            None,
            shape_ids,
        )
        
        super().__init__(components, print_traceback=print_traceback)
