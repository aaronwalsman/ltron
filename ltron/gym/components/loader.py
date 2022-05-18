import random

import numpy

from ltron.dataset.paths import get_tar_paths, get_dataset_info
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class DatasetLoaderComponent(LtronGymComponent):
    def __init__(self,
        scene_component,
        dataset,
        split,
        subset=None,
        rank=0,
        size=1,
        sample_mode='uniform',
    ):
        self.scene_component = scene_component
        self.dataset = dataset
        self.split = split
        self.subset = subset
        self.rank = rank
        self.size = size
        self.sample_mode = sample_mode
        self.dataset_info = get_dataset_info(self.dataset)
        
        self.tarfiles, self.dataset_paths = get_tar_paths(
            dataset, split, subset)
        self.length = len(self.dataset_paths)
        if sample_mode == 'uniform':
            self.dataset_ids = range(self.length)
        else:
            self.dataset_ids = range(self.rank, self.length, self.size)
        self.set_state({
            'finished':False,
            'episode_id':None,
            'dataset_id':None,
        })
    
    def reset(self):
        
        # clear the scene
        self.scene_component.clear_scene()
        
        # increment the episode id
        if self.episode_id is None:
            self.episode_id = 0
        else:
            self.episode_id += 1
        
        # pick the dataset id according to the sample_mode
        if self.sample_mode == 'uniform':
            self.dataset_id = random.choice(self.dataset_ids)
        elif self.sample_mode in ('sequential', 'multi_pass'):
            index = self.episode_id % len(self.dataset_ids)
            self.dataset_id = self.dataset_ids[index]
        elif self.sample_mode == 'single_pass':
            if self.episode_id < len(self.dataset_paths):
                self.dataset_id = self.dataset_ids[self.episode_id]
            else:
                self.finished = True
        else:
            raise ValueError('Unknown sample mode "%s"'%self.sample_mode)
        
        if not self.finished:
            #self.dataset_item = self.dataset_paths[self.dataset_id]
            #self.scene_component.brick_scene.import_ldraw(
            #    self.zipfile.open(self.dataset_paths[self.dataset_id]))
            tar_source, file_path = self.dataset_paths[self.dataset_id]
            text = self.tarfiles[tar_source].extractfile(file_path).read()
            self.scene_component.brick_scene.import_text(file_path, text)
        
        return None

    def step(self, action):
        return None, 0., False, None

    def get_state(self):
        state = {
            'finished':self.finished,
            'episode_id':self.episode_id,
            'dataset_id':self.dataset_id,
        }

        return state

    def set_state(self, state):
        self.finished = state['finished']
        self.episode_id = state['episode_id']
        self.dataset_id = state['dataset_id']
