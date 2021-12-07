import math
from collections import OrderedDict

import numpy

import gym
import gym.spaces as spaces
import numpy

from ltron.gym.envs.ltron_env import LtronEnv
from ltron.gym.components.scene import (
    EmptySceneComponent, DatasetSceneComponent)
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
        BreakAndMakePhaseSwitch, BreakAndMakeScore
)
from ltron.gym.components.assembly import AssemblyComponent
from ltron.gym.components.upright import UprightSceneComponent

'''
def break_and_make_template_action():
    return {
        'workspace_viewpoint' : 0,
        'handspace_viewpoint' : 0,
        'workspace_cursor' : {
            'activate':False,
            'position':numpy.array([0,0]),
            'polarity':0,
        },
        'handspace_cursor' : {
            'activate':False,
            'position':numpy.array([0,0]),
            'polarity':0,
        },
        'disassembly' : 0,
        'pick_and_place' : 0,
        'rotate' : 0,
        'insert_brick' : {
            'class_id' : 0,
            'color_id' : 0,
        },
        'phase_switch' : 0,
    }
'''

def break_and_make_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
    task='break_and_make',
    file_override=None,
    workspace_image_width=256,
    workspace_image_height=256,
    handspace_image_width=96,
    handspace_image_height=96,
    workspace_map_width=64,
    workspace_map_height=64,
    handspace_map_width=24,
    handspace_map_height=24,
    dataset_reset_mode='uniform',
    max_episode_length=32,
    workspace_render_args=None,
    handspace_render_args=None,
    randomize_viewpoint=True,
    randomize_colors=True,
    include_score=True,
    check_collisions=True,
    print_traceback=True,
    train=False,
):
    components = OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
        dataset, split, subset, rank, size, reset_mode=dataset_reset_mode)
    dataset_info = components['dataset'].dataset_info
    class_ids = dataset_info['class_ids']
    color_ids = dataset_info['color_ids']
    max_instances = dataset_info['max_instances_per_scene']
    max_edges = dataset_info['max_edges_per_scene']
    
    # scenes
    if file_override is not None:
        components['workspace_scene'] = SingleSceneComponent(
            file_override,
            class_ids,
            color_ids,
            max_instances,
            max_edges,
            render_args=workspace_render_args,
            track_snaps=True,
            collision_checker=check_collisions,
            #store_configuration=True,
        )
    else:
        components['workspace_scene'] = DatasetSceneComponent(
            dataset_component=components['dataset'],
            path_location=['mpd'],
            render_args=workspace_render_args,
            track_snaps=True,
            collision_checker=check_collisions,
            #store_configuration=True,
        )
    components['handspace_scene'] = EmptySceneComponent(
        class_ids=class_ids,
        color_ids=color_ids,
        max_instances=max_instances,
        max_edges=max_edges,
        render_args=handspace_render_args,
        track_snaps=True,
        collision_checker=False,
        #store_configuration=True,
    )
    
    # uprightify
    components['upright'] = UprightSceneComponent(
        scene_component = components['workspace_scene'])
    
    # max length
    components['max_length'] = MaxEpisodeLengthComponent(
        max_episode_length, observe_step=False)
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponent(
            dataset_info['color_ids'],
            components['workspace_scene'],
            randomize_frequency='reset',
        )
    
    # initial assembly
    components['initial_workspace_assembly'] = AssemblyComponent(
        components['workspace_scene'],
        class_ids,
        color_ids,
        max_instances,
        max_edges,
        update_frequency='reset',
        observe_assembly=train,
    )
    
    # viewpoint
    azimuth_steps = 8
    elevation_range = [math.radians(-30), math.radians(30)]
    elevation_steps = 2
    # TODO: make this correct
    workspace_distance_steps = 1
    workspace_distance_range=[320,320] # was 250
    handspace_distance_steps = 1
    handspace_distance_range=[180,180] # was 150
    if randomize_viewpoint:
        start_position='uniform'
    else:
        start_position=(0,0,0)
    components['workspace_viewpoint'] = ControlledAzimuthalViewpointComponent(
        components['workspace_scene'],
        azimuth_steps=azimuth_steps,
        elevation_range=elevation_range,
        elevation_steps=elevation_steps,
        distance_range=workspace_distance_range,
        distance_steps=workspace_distance_steps,
        aspect_ratio=workspace_image_width/workspace_image_height,
        start_position=start_position,
        auto_frame='reset',
        frame_button=True,
    )
    
    components['handspace_viewpoint'] = ControlledAzimuthalViewpointComponent(
        components['handspace_scene'],
        azimuth_steps=azimuth_steps,
        elevation_range=elevation_range,
        elevation_steps=elevation_steps,
        distance_range=handspace_distance_range,
        distance_steps=handspace_distance_steps,
        aspect_ratio=handspace_image_width/handspace_image_height,
        start_position=(0,0,0),
        auto_frame='none',
        frame_button=True
    )
    
    # utility rendering components
    workspace_pos_snap_render = SnapRenderComponent(
        workspace_map_width,
        workspace_map_height,
        components['workspace_scene'],
        polarity='+',
    )
    workspace_neg_snap_render = SnapRenderComponent(
        workspace_map_width,
        workspace_map_height,
        components['workspace_scene'],
        polarity='-',
    )
    if train:
        mask_render = SegmentationRenderComponent(
            workspace_map_width,
            workspace_map_height,
            components['workspace_scene'],
        )
    
    handspace_pos_snap_render = SnapRenderComponent(
        handspace_map_width,
        handspace_map_height,
        components['handspace_scene'],
        polarity='+',
    )
    handspace_neg_snap_render = SnapRenderComponent(
        handspace_map_width,
        handspace_map_height,
        components['handspace_scene'],
        polarity='-',
    )
    
    # cursors
    components['workspace_cursor'] = SnapCursor(
        max_instances,
        workspace_pos_snap_render,
        workspace_neg_snap_render,
        observe_instance_snap=train,
    )
    components['handspace_cursor'] = SnapCursor(
        max_instances,
        handspace_pos_snap_render,
        handspace_neg_snap_render,
        observe_instance_snap=train,
    )
    
    # action spaces
    components['rotate'] = CursorRotationAroundSnap(
        components['workspace_scene'],
        components['workspace_cursor'],
        check_collisions=check_collisions,
    )
    components['pick_and_place'] = CursorHandspacePickAndPlace(
        components['workspace_scene'],
        components['workspace_cursor'],
        components['handspace_scene'],
        components['handspace_cursor'],
        check_collisions=check_collisions,
    )
    components['disassembly'] = CursorDisassemblyComponent(
        max_instances,
        components['workspace_scene'],
        components['workspace_cursor'],
        handspace_component=components['handspace_scene'],
        check_collisions=check_collisions,
    )
    components['insert_brick'] = HandspaceBrickInserter(
        components['handspace_scene'],
        components['workspace_scene'],
        class_ids,
        color_ids,
        max_instances,
    )
    
    # phase_switch
    components['phase_switch'] = BreakAndMakePhaseSwitch(
        task=task,
        workspace_scene_component=components['workspace_scene'],
        workspace_viewpoint_component=components['workspace_viewpoint'],
        handspace_scene_component=components['handspace_scene'],
        dataset_component=components['dataset'],
        start_make_mode='clear',
        train=train,
    )
    
    # color render
    components['workspace_color_render'] = ColorRenderComponent(
        workspace_image_width,
        workspace_image_height,
        components['workspace_scene'],
        anti_alias=True,
    )
    
    components['handspace_color_render'] = ColorRenderComponent(
        handspace_image_width,
        handspace_image_height,
        components['handspace_scene'],
        anti_alias=True,
    )
    
    # snap render
    components['workspace_pos_snap_render'] = workspace_pos_snap_render
    components['workspace_neg_snap_render'] = workspace_neg_snap_render
    if train:
        components['workspace_mask_render'] = mask_render
    components['handspace_pos_snap_render'] = handspace_pos_snap_render
    components['handspace_neg_snap_render'] = handspace_neg_snap_render
    
    # current assembly
    components['workspace_assembly'] = AssemblyComponent(
        components['workspace_scene'],
        class_ids,
        color_ids,
        max_instances,
        max_edges,
        update_frequency = 'step',
        observe_assembly = train,
    )
    
    components['handspace_assembly'] = AssemblyComponent(
        components['handspace_scene'],
        class_ids,
        color_ids,
        max_instances,
        max_edges,
        update_frequency = 'step',
        observe_assembly = train,
    )
    
    # score
    if include_score:
        components['score'] = BreakAndMakeScore(
            components['initial_workspace_assembly'],
            components['workspace_assembly'],
            components['phase_switch'],
        )
    
    # build the env
    env = LtronEnv(components, print_traceback=print_traceback)
    
    return env
