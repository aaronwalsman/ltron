import random

from gym.spaces import Discrete, Dict

from brick_gym.dataset.paths import (
        get_dataset_paths, get_dataset_info, get_metadata)
import brick_gym.gym.spaces as bg_spaces
from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class DatasetPathComponent(BrickEnvComponent):
    def __init__(self,
            dataset,
            split,
            subset=None,
            rank=0,
            size=1,
            reset_mode='uniform'):
        
        self.set_state({'episode' : 0, 'scene_path' : None})
        
        self.reset_mode = reset_mode
        self.dataset = dataset
        self.split = split
        self.subset = subset
        self.dataset_info = get_dataset_info(self.dataset)
        self.dataset_paths = get_dataset_paths(
                self.dataset, self.split, self.subset, rank, size)
        
    def reset(self):
        if self.reset_mode == 'uniform':
            self.scene_path = random.choice(self.dataset_paths)
        elif self.reset_mode == 'sequential':
            self.scene_path = self.dataset_paths[
                    self.episode % len(self.dataset_paths)]
        elif self.reset_mode == 'single_pass':
            if self.episode < len(self.dataset_paths):
                self.scene_path = self.dataset_paths[self.episode]
            else:
                self.scene_path = None
        else:
            raise ValueError('Unknown reset mode "%s"'%self.reset_mode)
        if self.scene_path is not None:
            self.episode += 1
    
    def get_state(self):
        state = {'episode' : self.episode, 'scene_path' : self.scene_path}
       
    def set_state(self, state):
        self.episode = state['episode']
        self.scene_path = state['scene_path']
    
    def get_class_id(self, class_name):
        return self.dataset_info['class_ids'][class_name]
