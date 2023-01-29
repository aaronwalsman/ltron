import random

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
        self.subset = subset
        self.rank = 0
        self.size = 1
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.repeat = repeat
        
        self.initialized = False
        
        self.loaded_scenes = 0
        self.finished = False
    
    def reset(self, seed=None, rng=None, options=None):
        super().reset(seed=seed, rng=rng, options=options)
        
        if seed is not None or not self.initialized:
            random.seed(seed)
            #if seed is not None:
            #    dataset_rng = random.Random(seed)
            #else:
            #    dataset_rng = None
            
            self.dataset = get_mpd_webdataset(
                self.dataset_name,
                self.split,
                subset=self.subset,
                rank=self.rank,
                size=self.size,
                shuffle=self.shuffle,
                shuffle_buffer=self.shuffle_buffer,
                repeat=self.repeat,
            )
            self.iter = iter(self.dataset)
            self.initialized = True
            self.loaded_scenes = 0
        
        # clear the scene
        self.scene_component.clear_scene()
        
        if not self.finished:
            try:
                datapoint = next(self.iter)
                self.loaded_scenes += 1
            except StopIteration:
                self.finished = True
            else:
                text = datapoint['mpd']
                self.scene_component.brick_scene.import_text(
                    datapoint['__key__'] + '.mpd', text)
        
        return None, {}
    
    def get_state(self):
        print('WARNING: LOADER GET_STATE DOES NOT ACTUALLY WORK, NO RNG SAVED')
        state = {
            'loaded_scenes':self.loaded_scenes,
            'finished':self.finished,
        }
        
        return state
    
    def set_state(self, state):
        self.loaded_scenes = state['loaded_scenes']
        self.finished = state['finished']
        return None, {}
