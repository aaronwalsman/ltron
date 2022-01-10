import math

import ltron.settings as settings
from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.labels import (
        InstanceListComponent, InstanceGraphComponent)
from ltron.gym.components.render import (
        ColorRenderComponent, SegmentationRenderComponent)
from ltron.gym.components.viewpoint import (
        ControlledAzimuthalViewpointComponent,
        RandomizedAzimuthalViewpointComponent,
        FixedAzimuthalViewpointComponent)
from ltron.gym.components.visibility import (
        InstanceVisibilityComponent,
        PixelVisibilityComponent,
        InstanceRemovabilityComponent)
from ltron.gym.components.graph_tasks import InstanceGraphConstructionTask
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.random_floating_bricks import RandomFloatingBricks
from ltron.gym.components.random_floating_pairs import RandomFloatingPairs
from ltron.gym.components.spatial_info import BrickPosition

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
    num_classes = max(dataset_info['shape_ids'].values()) + 1
    
    # scene
    components['scene'] = SceneComponent(dataset_component = components['dataset'])
    
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
                list(dataset_info['shape_ids'].keys()),
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
    env = LtronEnv(components, print_traceback = print_traceback)
    
    print(env.action_space)
    print(env.observation_space)
    
    return env

def graph_supervision_env(
        dataset,
        split,
        subset=None,
        rank=0,
        size=1,
        augment_dataset=None,
        width=256,
        height=256,
        segmentation_width=None,
        segmentation_height=None,
        print_traceback=True,
        load_scenes=True,
        dataset_reset_mode='uniform',
        multi_hide=False,
        visibility_mode='instance',
        randomize_viewpoint=True,
        controlled_viewpoint=False,
        controlled_viewpoint_start_position='uniform',
        randomize_viewpoint_frequency='step',
        randomize_distance=True,
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
            augment_dataset=augment_dataset,
            reset_mode=dataset_reset_mode,
            observe_episode_id=True)
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    max_edges = dataset_info['max_edges_per_scene']
    if components['dataset'].augment_dataset is not None:
        augment_info = components['dataset'].augment_info
        max_instances = max(
                max_instances, augment_info['max_instances_per_scene'])
        max_edges = max(
                max_edges, augment_info['max_edges_per_scene'])
    
    num_classes = max(dataset_info['shape_ids'].values()) + 1
    
    # scene
    if load_scenes:
        path_component = components['dataset']
    else:
        path_component = None
    components['scene'] = SceneComponent(
            path_component=path_component, renderable=True)
    
    # random floating bricks
    if random_floating_bricks:
        if random_bricks_subset is None:
            random_bricks = list(dataset_info['shape_ids'].keys())
        else:
            random_bricks = list(
                    dataset_info['shape_ids'].keys())[:random_bricks_subset]
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
    
    # episode length
    components['episode_length'] = MaxEpisodeLengthComponent(max_instances)
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponent(
                dataset_info['all_colors'],
                components['scene'],
                randomize_frequency='step')
    
    # visiblity action space
    if visibility_mode == 'instance':
        components['visibility'] = InstanceVisibilityComponent(
                max_instances = max_instances,
                scene_component = components['scene'],
                multi = multi_hide,
                terminate_when_all_hidden = True)
    elif visibility_mode == 'pixel':
        # segmentation render
        if segmentation_width is None:
            segmentation_width = width
        if segmentation_height is None:
            segmentation_height = height
        components['pre_segmentation_render'] = SegmentationRenderComponent(
                segmentation_width, segmentation_height, components['scene'])
        components['visibility'] = PixelVisibilityComponent(
                width = width,
                height = height,
                scene_component = components['scene'],
                segmentation_component = components['pre_segmentation_render'],
                terminate_when_all_hidden = True)
    elif visibility_mode == 'multi_pixel':
        raise NotImplementedError
        components['visibility'] = MultiPixelVisibilityComponent(
                width = width,
                height = height,
                scene_component = components['scene'],
                terminate_when_all_hidden = True)
    
    # viewpoint
    if randomize_distance:
        distance = (0.8, 1.2)
    else:
        distance = (1.0, 1.0)
    if randomize_viewpoint:
        components['viewpoint'] = RandomizedAzimuthalViewpointComponent(
                components['scene'],
                distance = distance,
                #azimuth = math.radians(-135.),
                #elevation = math.radians(-30.),
                aspect_ratio = width/height,
                randomize_frequency=randomize_viewpoint_frequency)
    elif controlled_viewpoint:
        components['viewpoint'] = ControlledAzimuthalViewpointComponent(
                components['scene'],
                azimuth_steps = 24,
                elevation_range = (math.radians(-15), math.radians(-45)),
                elevation_steps = 4,
                distance_range = (200,350),
                distance_steps = 4,
                start_position = controlled_viewpoint_start_position)
    else:
        components['viewpoint'] = FixedAzimuthalViewpointComponent(
                components['scene'],
                distance = distance,
                azimuth = math.radians(-135.),
                elevation = math.radians(-30.),
                aspect_ratio = width/height)
    
    # random floating pairs
    if random_floating_pairs:
        augmentations = settings.datasets[dataset].replace(
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
    env = LtronEnv(components, print_traceback = print_traceback)
    
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
    num_classes = max(dataset_info['shape_ids'].values()) + 1
    
    # scene
    components['scene'] = SceneComponent(dataset_component = components['dataset'])
    
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
