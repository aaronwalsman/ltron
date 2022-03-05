import random

import numpy

from gym.spaces import Box, Discrete, Dict

from ltron.hierarchy import len_hierarchy, index_hierarchy, x_like_hierarchy
from ltron.dataset.paths import (
        get_dataset_paths, get_dataset_info, get_metadata)
import ltron.gym.spaces as bg_spaces
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class DatasetPathComponent(LtronGymComponent):
    def __init__(self,
        dataset,
        split,
        subset=None,
        rank=0,
        size=1,
        reset_mode='uniform',
        observe_episode_id=False,
        observe_dataset_id=False,
    ):
        
        self.dataset = dataset
        self.split = split
        self.subset = subset
        self.rank = rank
        self.size = size
        self.reset_mode = reset_mode
        self.dataset_info = get_dataset_info(self.dataset)
        self.observe_episode_id = observe_episode_id
        self.observe_dataset_id = observe_dataset_id
        
        self.dataset_paths = get_dataset_paths(dataset, split, subset)
        self.length = len_hierarchy(self.dataset_paths)
        #self.zipfile, self.names = get_zip_paths(dataset, split, subset=subset)
        #self.length = len(self.names)
        if reset_mode == 'uniform':
            self.dataset_ids = range(self.length)
        else:
            self.dataset_ids = range(self.rank, self.length, self.size)
        self.set_state({
            'initialized':False,
            'finished':False,
            'episode_id':None,
            'dataset_id':None,
        })
        
        observation_space = {}
        if self.observe_episode_id:
            observation_space['episode_id'] = Box(
                low=0, high=numpy.inf, shape=(1,))
        if self.observe_dataset_id:
            observation_space['dataset_id'] = Discrete(
                len(self.dataset_paths))
                #len(self.names))
        if len(observation_space):
            self.observation_space = Dict(observation_space)
    
    def observe(self):
        #assert self.initialized
        self.observation = {}
        if self.observe_dataset_id:
            self.observation['dataset_id'] = self.dataset_id
        if self.observe_episode_id:
            self.observation['episode_id'] = self.episode_id
    
    def reset(self):
        # three cases:
        # 1. hasn't been initialized yet
        # 2. normal operation
        # 3. finished (single pass only)
        
        # increment episode_id if initialized, otherwise initialize
        if self.initialized:
            self.episode_id += 1
        else:
            self.initialized = True
            self.episode_id = 0
            self.dataset_id = 0
        
        # pick the dataset id according to the reset_mode
        if self.reset_mode == 'uniform':
            self.dataset_id = random.choice(self.dataset_ids)
        elif (self.reset_mode == 'sequential' or
            self.reset_mode == 'multi_pass'
        ):
            index = self.episode_id % len(self.dataset_ids)
            self.dataset_id = self.dataset_ids[index]
        elif self.reset_mode == 'single_pass':
            if self.episode_id < len(self.dataset_paths['mpd']):
            #if self.episode_id < len(self.names):
                self.dataset_id = self.dataset_ids[self.episode_id]
            else:
                self.finished = True
        else:
            raise ValueError('Unknown reset mode "%s"'%self.reset_mode)
        
        if not self.finished:
            self.dataset_item = index_hierarchy(
                self.dataset_paths, self.dataset_id)
            #self.dataset_item = self.names[self.dataset_id]
        
        self.observe()
        return self.observation
    
    def step(self, action):
        self.observe()
        return self.observation, 0., False, None
    
    def get_state(self):
        state = {
            'initialized':self.initialized,
            'finished':self.finished,
            'episode_id':self.episode_id,
            'dataset_id':self.dataset_id,
        }
        
        return state
       
    def set_state(self, state):
        self.initialized = state['initialized']
        self.finished = state['finished']
        self.episode_id = state['episode_id']
        self.dataset_id = state['dataset_id']
        
        self.observe()
        return self.observation
