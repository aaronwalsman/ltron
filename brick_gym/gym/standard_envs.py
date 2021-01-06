import math

from brick_gym.envs.brick_env import BrickEnv
from brick_gym.envs.components.episode import MaxEpisodeLengthComponent
from brick_gym.envs.components.dataset import DatasetComponent
from brick_gym.envs.components.labels import GraphLabelComponent
from brick_gym.envs.components.render import (
        RendererComponent, ColorRenderComponent, MaskRenderComponent)
from brick_gym.envs.components.viewpoint import FixedAzimuthalViewpointComponent
from brick_gym.envs.components.visibility import InstanceVisibilityComponent
from brick_gym.envs.components.graph_tasks import GraphReconstructionTask

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
    mask_component = MaskRenderComponent(height, width)
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
        height=256,
        width=256,
        *args,
        **kwargs):
    
    components = []
    
    # dataset
    dataset_component = DatasetComponent(dataset, split, subset, rank, size)
    components.append(dataset_component)
    max_instances = dataset_component.dataset_info['max_instances_per_scene']
    num_classes = max(dataset_component.dataset_info['class_ids'].values())+1
    
    # max length episodes
    fixed_length_episode_component = MaxEpisodeLengthComponent(max_instances)
    components.append(fixed_length_episode_component)
    
    # training labels
    if train:
        graph_label_component = GraphLabelComponent(num_classes, max_instances)
        components.append(graph_label_component)
    
    # visiblity action space
    visibility_component = InstanceVisibilityComponent(
            max_instances, terminate_when_all_hidden=True)
    components.append(visibility_component)
    
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
    mask_component = MaskRenderComponent(height, width)
    components.append(mask_component)
    
    # graph reconstruction evaluation score
    graph_task = GraphReconstructionTask(num_classes, max_instances)
    components.append(graph_task)
    
    # build the env
    env = BrickEnv(components, *args, **kwargs)
    
    return env
