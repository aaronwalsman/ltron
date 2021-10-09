import math
from collections import OrderedDict

import numpy

import gym
import gym.spaces as spaces
import numpy

from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.render import (
        ColorRenderComponent, SegmentationRenderComponent, SnapRenderComponent)
from ltron.gym.components.disassembly import PixelDisassemblyComponent
from ltron.gym.components.rotation import RotationAroundSnap
from ltron.gym.components.pick_and_place import (
        PickAndPlace, HandspacePickAndPlace)
from ltron.gym.components.brick_inserter import HandspaceBrickInserter
from ltron.gym.components.viewpoint import (
        ControlledAzimuthalViewpointComponent)
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.reassembly import Reassembly


def reassembly_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
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
    #workspace_render_args=None,
    #handspace_render_args=None,
    randomize_viewpoint=True,
    randomize_colors=True,
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
    components['workspace_scene'] = SceneComponent(
        dataset_component=components['dataset'],
        path_location=['mpd'],
        render_args=workspace_render_args,
        track_snaps=True,
        collision_checker=check_collisions,
    )
    
    components['handspace_scene'] = SceneComponent(
        render_args=handspace_render_args,
        track_snaps=True,
        collision_checker=False,
    )
    
    components['max_length'] = MaxEpisodeLengthComponent(
        max_episode_length, observe_step=False)
    
    # cursors
    components['workspace_cursor'] = Cursor(
        (workspace_map_height, workspace_map_width, 2))
    components['handspace_cursor'] = Cursor(
        (handspace_map_height, handspace_map_width, 2))
    
    # viewpoint
    azimuth_steps = 8
    elevation_range = [math.radians(-30), math.radians(30)]
    elevation_steps = 2
    # TODO: make this correct
    workspace_distance_steps = 1
    workspace_distance_range=[250,250]
    handspace_distance_steps = 1
    handspace_distance_range=[150,150]
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
        distance_steps=distance_steps,
        aspect_ratio=workspace_image_width/workspace_image_height,
        start_position=start_position,
        frame_scene=True,
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
        frame_scene=True,
    )
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponent(
            dataset_info['color_ids'],
            components['workspace_scene'],
            randomize_frequency='reset',
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
    
    # action spaces
    components['disassembly'] = CursorDisassemblyComponent(
        max_instances,
        components['workspace_scene'],
        componetns['workspace_cursor'],
        workspace_pos_snap_render,
        workspace_neg_snap_render,
        handspace_component=components['handspace_scene'],
        check_collisions=check_collisions,
    )
    components['rotate'] = RotationAroundSnap(
        components['workspace_scene'],
        workspace_pos_snap_render,
        workspace_neg_snap_render,
        check_collisions=check_collisions,
    )
    components['pick_and_place'] = HandspacePickAndPlace(
        components['workspace_scene'],
        workspace_pos_snap_render,
        workspace_neg_snap_render,
        components['handspace_scene'],
        handspace_pos_snap_render,
        handspace_neg_snap_render,
        check_collisions=check_collisions,
    )
    components['insert_brick'] = HandspaceBrickInserter(
        components['handspace_scene'],
        components['workspace_scene'],
        class_ids,
        color_ids,
        max_instances,
    )
    
    # reassembly
    components['reassembly'] = Reassembly(
        class_ids=class_ids,
        color_ids=color_ids,
        max_instances=max_instances,
        max_edges=max_edges,
        max_snaps_per_brick=max_snaps,
        workspace_scene_component=components['workspace_scene'],
        workspace_viewpoint_component=components['workspace_viewpoint'],
        handspace_scene_component=components['handspace_scene'],
        dataset_component=components['dataset'],
        reassembly_mode='clear',
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
    
    if train:
        components['workspace_segmentation_render'] = (
             SegmentationRenderComponent(
                workspace_map_width,
                workspace_map_height,
                components['workspace_scene'],
            )
        )
    
    # snap render
    components['workspace_pos_snap_render'] = workspace_pos_snap_render
    components['workspace_neg_snap_render'] = workspace_neg_snap_render
    components['handspace_pos_snap_render'] = handspace_pos_snap_render
    components['handspace_neg_snap_render'] = handspace_neg_snap_render
    
    # build the env
    env = LtronEnv(components, print_traceback=print_traceback)
    
    return env

if __name__ == '__main__':
    #interactive_env = InteractiveReassemblyEnv(
    interactive_env = InteractiveHandspaceReassemblyEnv(
        dataset='random_six',
        split='simple_single',
        subset=1,
        train=True,
        randomize_colors=False,
        randomize_viewpoint=False)
    interactive_env.start()
