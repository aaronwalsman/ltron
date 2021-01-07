import math
import collections

from brick_gym.gym.brick_env import BrickEnv
from brick_gym.gym.components.episode import MaxEpisodeLengthComponent
from brick_gym.gym.components.dataset import DatasetPathComponent
from brick_gym.gym.components.labels import GraphLabelComponent
from brick_gym.gym.components.render import (
        ColorRenderComponent, SegmentationRenderComponent)
from brick_gym.gym.components.viewpoint import FixedAzimuthalViewpointComponent
from brick_gym.gym.components.visibility import InstanceVisibilityComponent
from brick_gym.gym.components.graph_tasks import GraphReconstructionTask

def dummy_env():
    
    components = []
    
    # episode
    fixed_length_episode_component = MaxEpisodeLengthComponent(1)
    components.append(fixed_length_episode_component)
    
    # dataset
    dataset_component = DatasetComponent('random_stack', 'train_mpd')
    height = 256
    width = 256
    components.append(dataset_component)
    
    # renderer
    renderer_component = RendererComponent()
    components.append(renderer_component)
    # viewpoint
    viewpoint_component = FixedAzimuthalViewpointComponent(
            azimuth = math.radians(30.),
            elevation = math.radians(-45.))
    # color render
    color_component = ColorRenderComponent(height, width)
    components.append(color_component)
    # mask render
    segmentation_component = SegmentationRenderComponent(height, width)
    components.append(mask_component)
    
    env = BrickEnv(components)
    
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
        *args,
        **kwargs):
    
    components = collections.OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
            dataset, split, subset, rank, size, reset_mode='unifom')
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
    
    # mask render
    components['mask_render'] = SegmentationRenderComponent(
            width, height, components['scene'])
    
    # visiblity action space
    components['visibility'] = InstanceVisibilityComponent(
            scene_component = components['scene'],
            terminate_when_all_hidden = True)
    
    # viewpoint
    components['viewpoint'] = FixedAzimuthalViewpointComponent(
            azimuth = math.radians(30.),
            elevation = math.radians(-45.),
            aspect_ratio = width/height)
    
    # graph reconstruction evaluation score
    components['task'] = GraphReconstructionTask(num_classes, max_instances)
    
    # build the env
    env = BrickEnv(components, *args, **kwargs)
    
    return env
