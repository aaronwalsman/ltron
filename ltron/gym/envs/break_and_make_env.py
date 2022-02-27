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
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.render import (
        ColorRenderComponent, SegmentationRenderComponent, SnapRenderComponent)
from ltron.gym.components.cursor import SnapCursor
from ltron.gym.components.disassembly import CursorDisassemblyComponent
from ltron.gym.components.rotation import CursorRotationAroundSnap
from ltron.gym.components.pick_and_place import (
        CursorHandspacePickAndPlace)
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

# TODO: Needs a config

class BreakAndMakeEnvConfig(Config):
    dataset = 'random_construction_6_6'
    ldraw_file = None
    split = 'train'
    subset = None
    
    task = 'break_and_make'
    
    table_image_height = 256
    table_image_width = 256
    hand_image_height = 96
    hand_image_width = 96
    
    table_map_height = 64
    table_map_width = 64
    hand_map_height = 24
    hand_map_width = 24
    
    dataset_reset_mode = 'uniform'
    
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
    
    observe_dataset_id = False
    
    allow_snap_flip = False

#def break_and_make_env(config, rank, size):
#    dataset,
#    split,
#    subset=None,
#    rank=0,
#    size=1,
#    task='break_and_make',
#    file_override=None,
#    workspace_image_width=256,
#    workspace_image_height=256,
#    handspace_image_width=96,
#    handspace_image_height=96,
#    workspace_map_width=64,
#    workspace_map_height=64,
#    handspace_map_width=24,
#    handspace_map_height=24,
#    dataset_reset_mode='uniform',
#    max_episode_length=32,
#    workspace_render_args=None,
#    handspace_render_args=None,
#    randomize_viewpoint=True,
#    randomize_colors=True,
#    include_score=True,
#    check_collision=True,
#    print_traceback=True,
#    train=False,
#):

class BreakAndMakeEnv(LtronEnv):
    def __init__(self, config, rank=0, size=1, print_traceback=False):
        components = OrderedDict()
        
        # dataset
        components['dataset'] = DatasetPathComponent(
            config.dataset,
            config.split,
            config.subset,
            rank,
            size,
            reset_mode=config.dataset_reset_mode,
            observe_dataset_id = config.observe_dataset_id,
        )
        dataset_info = components['dataset'].dataset_info
        shape_ids = dataset_info['shape_ids']
        color_ids = dataset_info['color_ids']
        max_instances = dataset_info['max_instances_per_scene']
        max_edges = dataset_info['max_edges_per_scene']
        
        # scenes
        if config.ldraw_file is not None:
            components['table_scene'] = SingleSceneComponent(
                config.ldraw_file,
                shape_ids,
                color_ids,
                max_instances,
                max_edges,
                track_snaps=True,
                collision_checker=config.check_collision,
                #render_args={'egl_device':config.egl_device},
            )
        else:
            components['table_scene'] = DatasetSceneComponent(
                dataset_component=components['dataset'],
                path_location=['mpd'],
                track_snaps=True,
                collision_checker=config.check_collision,
                #render_args={'egl_device':config.egl_device},
            )
        components['hand_scene'] = EmptySceneComponent(
            shape_ids=shape_ids,
            color_ids=color_ids,
            max_instances=max_instances,
            max_edges=max_edges,
            #render_args=hand_render_args,
            track_snaps=True,
            collision_checker=False,
        )
        
        # uprightify
        components['upright'] = UprightSceneComponent(
            scene_component = components['table_scene'])
        
        # max length
        components['step'] = MaxEpisodeLengthComponent(
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
        
        # viewpoint
        azimuth_steps = 8
        elevation_range = [math.radians(-30), math.radians(30)]
        elevation_steps = 2
        # TODO: make this correct
        table_distance_steps = 1
        table_distance_range=[320,320] # was 250
        hand_distance_steps = 1
        hand_distance_range=[180,180] # was 150
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
        )
        table_neg_snap_render = SnapRenderComponent(
            config.table_map_width,
            config.table_map_height,
            components['table_scene'],
            polarity='-',
        )
        if config.train:
            mask_render = SegmentationRenderComponent(
                config.table_map_width,
                config.table_map_height,
                components['table_scene'],
            )
        
        hand_pos_snap_render = SnapRenderComponent(
            config.hand_map_width,
            config.hand_map_height,
            components['hand_scene'],
            polarity='+',
        )
        hand_neg_snap_render = SnapRenderComponent(
            config.hand_map_width,
            config.hand_map_height,
            components['hand_scene'],
            polarity='-',
        )
        
        # cursors
        components['table_cursor'] = SnapCursor(
            max_instances,
            table_pos_snap_render,
            table_neg_snap_render,
            observe_instance_snap=config.train,
        )
        components['hand_cursor'] = SnapCursor(
            max_instances,
            hand_pos_snap_render,
            hand_neg_snap_render,
            observe_instance_snap=config.train,
        )
        
        # action spaces
        components['rotate'] = CursorRotationAroundSnap(
            components['table_scene'],
            components['table_cursor'],
            check_collision=config.check_collision,
            allow_snap_flip=config.allow_snap_flip,
        )
        components['pick_and_place'] = CursorHandspacePickAndPlace(
            components['table_scene'],
            components['table_cursor'],
            components['hand_scene'],
            components['hand_cursor'],
            check_collision=config.check_collision,
        )
        components['disassembly'] = CursorDisassemblyComponent(
            max_instances,
            components['table_scene'],
            components['table_cursor'],
            hand_scene_component=components['hand_scene'],
            check_collision=config.check_collision,
        )
        components['insert_brick'] = HandspaceBrickInserter(
            components['hand_scene'],
            components['table_scene'],
            shape_ids,
            color_ids,
            max_instances,
        )
        
        # phase
        if config.task == 'break_and_make':
            components['phase'] = BreakAndMakePhaseSwitch(
                table_scene_component=components['table_scene'],
                table_viewpoint_component=components['table_viewpoint'],
                hand_scene_component=components['hand_scene'],
                dataset_component=components['dataset'],
                start_make_mode='clear',
                train=config.train,
            )
        elif config.task == 'break_only':
            components['phase'] = BreakOnlyPhaseSwitch()
        
        # color render
        components['table_color_render'] = ColorRenderComponent(
            config.table_image_width,
            config.table_image_height,
            components['table_scene'],
            anti_alias=True,
        )
        
        components['hand_color_render'] = ColorRenderComponent(
            config.hand_image_width,
            config.hand_image_height,
            components['hand_scene'],
            anti_alias=True,
        )
        
        # tile
        if config.tile_color_render:
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
        
        # snap render
        components['table_pos_snap_render'] = table_pos_snap_render
        components['table_neg_snap_render'] = table_neg_snap_render
        if config.train:
            components['table_mask_render'] = mask_render
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
        if config.include_score:
            if config.task == 'break_and_make':
                components['score'] = BreakAndMakeScore(
                    components['initial_table_assembly'],
                    components['table_assembly'],
                    components['phase'],
                    shape_ids,
                )
            else:
                components['score'] = BreakOnlyScore(
                    components['initial_table_assembly'],
                    components['table_assembly'],
                )
        
        # build the env
        #env = LtronEnv(components, print_traceback=print_traceback)
        
        super().__init__(components, print_traceback=print_traceback)
