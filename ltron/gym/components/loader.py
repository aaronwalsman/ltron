import random

import numpy

from supermecha import SuperMechaComponent

from ltron.dataset.webdataset import get_mpd_webdataset

class DatasetLoader(SuperMechaComponent):
    def __init__(self,
        scene_component,
        dataset_name,
        split,
        subset=None,
        rank=0,
        size=1,
        shuffle=False,
        shuffle_buffer=100,
        repeat=False,
    ):
        self.scene_component = scene_component
        self.dataset_name = dataset_name
        self.split = split
        
        self.dataset = get_mpd_webdataset(
            self.dataset_name,
            self.split,
            subset=subset,
            rank=rank,
            size=size,
            shuffle=shuffle,
            shuffle_buffer=shuffle_buffer,
            repeat=repeat,
        )
        self.iter = iter(self.dataset)
        
        self.set_state({
            'finished':False,
        })
    
    def reset(self, seed=None, rng=None, options=None):
        super().reset(seed=seed, rng=rng, options=options)
        
        # clear the scene
        self.scene_component.clear_scene()
        
        if not self.finished:
            try:
                datapoint = next(self.iter)
            except StopIteration:
                self.finished = True
            else:
                text = datapoint['mpd']
                self.scene_component.brick_scene.import_text(
                    datapoint['__key__'] + '.mpd', text)
        
        return None, None

    def get_state(self):
        state = {
            'finished':self.finished,
        }

        return state

    def set_state(self, state):
        self.finished = state['finished']
