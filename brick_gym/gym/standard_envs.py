import math
import collections

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
from brick_gym.gym.components.visibility import InstanceVisibilityComponent
from brick_gym.gym.components.graph_tasks import InstanceGraphConstructionTask
from brick_gym.gym.components.colors import RandomizeColorsComponent

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
        randomize_colors=True):
    
    components = collections.OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
            dataset, split, subset, rank, size, reset_mode=dataset_reset_mode)
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    num_classes = max(dataset_info['class_ids'].values()) + 1
    
    # scene
    components['scene'] = SceneComponent(path_component = components['dataset'])
    
    # episode length
    components['episode_length'] = MaxEpisodeLengthComponent(max_instances)
    
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
        print_traceback=True,
        dataset_reset_mode='uniform',
        randomize_viewpoint=True,
        randomize_viewpoint_frequency='step',
        randomize_colors=True):
    
    components = collections.OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
            dataset, split, subset, rank, size, reset_mode=dataset_reset_mode)
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    num_classes = max(dataset_info['class_ids'].values()) + 1
    
    # scene
    components['scene'] = SceneComponent(path_component = components['dataset'])
    
    # episode length
    components['episode_length'] = MaxEpisodeLengthComponent(max_instances)
    
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
    
    # graph labels
    components['graph_label'] = InstanceGraphComponent(
            num_classes,
            max_instances,
            max_instances**2,
            components['dataset'],
            components['scene'])
    
    # graph_task
    components['graph_task'] = InstanceGraphConstructionTask(
            num_classes,
            max_instances,
            max_instances**2,
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
