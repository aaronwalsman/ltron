import random

from brick_gym.dataset.paths import (
        get_dataset_paths, get_dataset_info, get_metadata)
import brick_gym.gym.spaces as bg_spaces
from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class DatasetComponent(BrickEnvComponent):
    def __init__(self,
            dataset,
            split,
            subset=None,
            rank=0,
            size=1,
            scene_path_key='scene_path',
            scene_metadata_key='scene_metadata',
            reset_mode='uniform'):
        
        self.dataset_info = get_dataset_info(dataset)
        self.dataset_paths = get_dataset_paths(
                dataset, split, subset, rank, size)
        
        self.scene_path_key = scene_path_key
        self.scene_metadata_key = scene_metadata_key
        
        self.reset_mode = reset_mode
    
    def initialize_state(self, state):
        self.episode = 0
        if self.scene_path_key is not None:
            state[self.scene_path_key] = None
        if self.scene_metadata_key is not None:
            state[self.scene_metadata_key] = None
    
    def reset_state(self, state):
        if self.reset_mode == 'uniform':
            path = random.choice(self.dataset_paths)
        elif self.reset_mode == 'sequential':
            path = self.dataset_paths[self.episode % len(self.dataset_paths)]
        if self.scene_path_key is not None:
            state[self.scene_path_key] = path
        if self.scene_metadata_key is not None:
            metadata = get_metadata(path)
            state[self.scene_metadata_key] = metadata
        self.episode += 1
