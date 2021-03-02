import math
import collections

import brick_gym.config as config
from brick_gym.gym.brick_env import BrickEnv
from brick_gym.gym.components.scene import SceneComponent
from brick_gym.gym.components.episode import MaxEpisodeLengthComponent
from brick_gym.gym.components.dataset import DatasetPathComponent
from brick_gym.gym.components.labels import (
        InstanceListComponent, InstanceGraphComponent)
from brick_gym.gym.components.render import (
        ColorRenderComponent, SegmentationRenderComponent)
from brick_gym.gym.components.viewpoint import (
        RandomizedAzimuthalViewpointComponent, FixedAzimuthalViewpointComponent)
from brick_gym.gym.components.visibility import (
        InstanceVisibilityComponent, InstanceRemovabilityComponent)
from brick_gym.gym.components.graph_tasks import InstanceGraphConstructionTask
from brick_gym.gym.components.colors import RandomizeColorsComponent
from brick_gym.gym.components.random_floating_bricks import RandomFloatingBricks
from brick_gym.gym.components.random_floating_pairs import RandomFloatingPairs
from brick_gym.gym.components.spatial_info import BrickPosition

def segmentation_supervision_env(
        dataset,
        split,
        subset=None,
        rank=0,
        size=1,
        width=256,
        height=256,
        print_traceback=True,
        dataset_reset_mode='uniform',
        randomize_viewpoint=True,
        randomize_viewpoint_frequency='step',
        randomize_colors=True,
        random_floating_bricks=True):
    
    components = collections.OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
            dataset, split, subset, rank, size, reset_mode=dataset_reset_mode)
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    num_classes = max(dataset_info['class_ids'].values()) + 1
    
    # scene
    components['scene'] = SceneComponent(path_component = components['dataset'])
    
    # viewpoint
    if randomize_viewpoint:
        components['viewpoint'] = RandomizedAzimuthalViewpointComponent(
                components['scene'],
                #azimuth = math.radians(-135.),
                #elevation = math.radians(-30.),
                #tilt = (math.radians(-45.0), math.radians(45.0)),
                aspect_ratio = width/height,
                randomize_frequency=randomize_viewpoint_frequency)
    else:
        components['viewpoint'] = FixedAzimuthalViewpointComponent(
                components['scene'],
                azimuth = math.radians(-135.),
                elevation = math.radians(-30.),
                aspect_ratio = width/height)
    
    # random floating bricks
    if random_floating_bricks:
        components['random_floating_bricks'] = RandomFloatingBricks(
                components['scene'],
                list(dataset_info['class_ids'].keys()),
                dataset_info['all_colors'])
        max_instances += (
                components['random_floating_bricks'].bricks_per_scene[-1])
    
    # episode length
    components['episode_length'] = MaxEpisodeLengthComponent(max_instances)
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponent(
                dataset_info['all_colors'],
                components['scene'],
                randomize_frequency='step')
    
    # visiblity action space
    components['visibility'] = InstanceVisibilityComponent(
            max_instances = max_instances,
            scene_component = components['scene'],
            terminate_when_all_hidden = True)
    
    # color render
    components['color_render'] = ColorRenderComponent(
            width, height, components['scene'], anti_alias=True)
    
    # segmentation render
    components['segmentation_render'] = SegmentationRenderComponent(
            width, height, components['scene'])
    
    # node labels
    components['instance_labels'] = InstanceListComponent(
            num_classes,
            max_instances,
            components['dataset'],
            components['scene'])
    
    # build the env
    env = BrickEnv(components, print_traceback = print_traceback)
    
    print(env.action_space)
    print(env.observation_space)
    
    return env

def graph_supervision_env(
        dataset,
        split,
        subset=None,
        rank=0,
        size=1,
        width=256,
        height=256,
        segmentation_width=None,
        segmentation_height=None,
        print_traceback=True,
        load_scenes=True,
        dataset_reset_mode='uniform',
        multi_hide=False,
        randomize_viewpoint=True,
        randomize_viewpoint_frequency='step',
        randomize_colors=True,
        random_floating_bricks=False,
        random_floating_pairs=False,
        random_bricks_per_scene=(10,20),
        random_bricks_subset=None,
        random_bricks_rotation_mode='local_identity'):
    
    components = collections.OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
            dataset,
            split,
            subset,
            rank,
            size,
            reset_mode=dataset_reset_mode,
            observe_episode_id=True)
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    max_edges = dataset_info['max_edges_per_scene']
    
    num_classes = max(dataset_info['class_ids'].values()) + 1
    
    # scene
    if load_scenes:
        path_component = components['dataset']
    else:
        path_component = None
    components['scene'] = SceneComponent(path_component=path_component)
    
    # random floating bricks
    if random_floating_bricks:
        if random_bricks_subset is None:
            random_bricks = list(dataset_info['class_ids'].keys())
        else:
            random_bricks = list(
                    dataset_info['class_ids'].keys())[:random_bricks_subset]
        components['random_floating_bricks'] = RandomFloatingBricks(
                components['scene'],
                random_bricks,
                dataset_info['all_colors'],
                bricks_per_scene = random_bricks_per_scene,
                rotation_mode = random_bricks_rotation_mode)
        max_instances += (
                components['random_floating_bricks'].bricks_per_scene[-1])
        # this shouldn't generate more edges, but maybe it accidentally
        # does sometimes?
        max_edges += (
                components['random_floating_bricks'].bricks_per_scene[-1])
    
    # viewpoint
    if randomize_viewpoint:
        components['viewpoint'] = RandomizedAzimuthalViewpointComponent(
                components['scene'],
                #azimuth = math.radians(-135.),
                #elevation = math.radians(-30.),
                aspect_ratio = width/height,
                randomize_frequency=randomize_viewpoint_frequency)
    else:
        components['viewpoint'] = FixedAzimuthalViewpointComponent(
                components['scene'],
                azimuth = math.radians(-135.),
                elevation = math.radians(-30.),
                aspect_ratio = width/height)
    
    # random floating pairs
    if random_floating_pairs:
        augmentations = config.datasets[dataset].replace(
                '.json', '_edge_augmentations.json')
        components['random_floating_pairs'] = RandomFloatingPairs(
                components['scene'],
                augmentations,
                dataset_info['all_colors'],
                pairs_per_scene = random_bricks_per_scene,
                rotation_mode = random_bricks_rotation_mode)
        max_instances += (
                components['random_floating_pairs'].pairs_per_scene[-1]*2)
        max_edges += (
                components['random_floating_pairs'].pairs_per_scene[-1]*2)
    
    # episode length
    components['episode_length'] = MaxEpisodeLengthComponent(max_instances)
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponent(
                dataset_info['all_colors'],
                components['scene'],
                randomize_frequency='step')
    
    # visiblity action space
    components['visibility'] = InstanceVisibilityComponent(
            max_instances = max_instances,
            scene_component = components['scene'],
            multi = multi_hide,
            terminate_when_all_hidden = True)
    
    # brick height (TEMP)
    components['brick_position'] = BrickPosition(
            max_instances,
            components['scene'])
    components['removability'] = InstanceRemovabilityComponent(
            max_instances,
            components['scene'])
    
    # color render
    components['color_render'] = ColorRenderComponent(
            width, height, components['scene'], anti_alias=True)
    
    # segmentation render
    if segmentation_width is None:
        segmentation_width = width
    if segmentation_height is None:
        segmentation_height = height
    components['segmentation_render'] = SegmentationRenderComponent(
            segmentation_width, segmentation_height, components['scene'])
    
    # graph labels
    components['graph_label'] = InstanceGraphComponent(
            num_classes,
            max_instances,
            max_edges,
            components['dataset'],
            components['scene'])
    
    # graph_task
    components['graph_task'] = InstanceGraphConstructionTask(
            num_classes,
            max_instances*2,
            max_edges*8,
            components['scene'],
            components['dataset'])
    
    # build the env
    env = BrickEnv(components, print_traceback = print_traceback)
    
    return env


def graph_env(
        dataset,
        split,
        subset=None,
        rank=0,
        size=1,
        train=False,
        width=256,
        height=256,
        print_traceback=True):
    
    components = collections.OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
            dataset, split, subset, rank, size, reset_mode='uniform')
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    num_classes = max(dataset_info['class_ids'].values()) + 1
    
    # scene
    components['scene'] = SceneComponent(path_component = components['dataset'])
    
    # training labels
    if train:
        components['graph_label'] = GraphLabelComponent(
                num_classes,
                max_instances,
                components['scene'])
    
    # color render
    components['color_render'] = ColorRenderComponent(
            width, height, components['scene'], anti_alias=True)
    
    # segmentation render
    components['segmentation_render'] = SegmentationRenderComponent(
            width, height, components['scene'])
    
    # visiblity action space
    components['visibility'] = InstanceVisibilityComponent(
            max_instances = max_instances,
            scene_component = components['scene'],
            terminate_when_all_hidden = True)
    
    # viewpoint
    components['viewpoint'] = FixedAzimuthalViewpointComponent(
            components['scene'],
            azimuth = math.radians(30.),
            elevation = math.radians(-45.),
            aspect_ratio = width/height)
    
    # graph reconstruction evaluation score
    components['task'] = GraphConstructionTask(
            num_classes, max_instances, components['scene'])
    
    # build the env
    env = BrickEnv(components, print_traceback = print_traceback)
    
    return env
