import math

from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.render import (
        ColorRenderComponent, SegmentationRenderComponent)
from ltron.gym.components.viewpoint import (
        FixedAzimuthalViewpointComponent)
from ltron.gym.components.colors import RandomizeColorsComponent

def reassembly_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
    width=256,
    height=256,
    print_traceback=True,
    dataset_reset_mode='uniform',
    randomize_colors=True
):
    
    components = collections.OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
        dataset, split, subset, rank, size, reset_mode=dataset_reset_mode)
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    
    # scene
    components['scene'] = SceneComponent(
        dataset_component=components['dataset'])
    
    # viewpoint
    components['viewpoint'] = FixedAzimuthalViewpointComponent(
        components['scene'],
        azimuth=math.radians(-135.),
        elevation=math.radians(-30.),
        aspect_ratio=width/height,
    )
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponents(
            dataset_info['all_colors'],
            components['scene'],
            randomize_frequency='scene',
        )
    
    # action space goes here
    
    # scoring goes here
    
    # color render
    components['color_render'] = ColorRenderComponent(
        width, height, components['scene'], anti_alias=True)
    
    # snap render
    components['pos_snap_render'] = SnapRenderComponent(
        width, height, components['scene'], polarity='+')
    components['neg_snap_render'] = SnapRenderComponent(
        width, height, components['scene'], polarity='-')
    
    # reward
    components['reconstruction_score'] = ReconstructionScore(
        components['scene'])
    
    # build the env
    env = BrickEnv(components, print_traceback=print_traceback)
    
    return env
