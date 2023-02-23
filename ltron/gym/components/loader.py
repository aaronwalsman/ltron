import random

from steadfast.config import Config

from supermecha import SuperMechaComponent

from ltron.dataset.webdataset import get_mpd_webdataset

class ClearScene(SuperMechaComponent):
    def __init__(self, scene_component):
        self.scene_component = scene_component
    
    def reset(self, seed=None, options=None):
        super().reset(seed)
        self.scene_component.clear_scene()
        
        return None, {}

class SingleSceneLoader(SuperMechaComponent):
    def __init__(self, scene_component, file_path):
        self.scene_component = scene_component
        self.file_path = file_path
    
    def reset(self, seed=None, options=None):
        super().reset(seed)
        self.scene_component.clear_scene()
        self.scene_component.brick_scene.import_ldraw(self.file_path)
        
        return None, {}

class DatasetLoader(SuperMechaComponent):
    def __init__(self,
        scene_component,
        dataset_name,
        split,
        subset=None,
        rank=0,
        size=1,
        shuffle=False,
        shuffle_buffer=1000,
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
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        if seed is not None or not self.initialized:
            random.seed(seed)
            
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

class LoaderConfig(Config):
    load_scene = None
    train_dataset = None
    train_split = None
    train_subset = None
    train_repeat = 1
    train_shuffle = True
    eval_dataset = None
    eval_split = None
    eval_subset = None
    eval_repeat = 1
    eval_shuffle = False

def make_loader(config, scene_component, train=False, load_key='load_scene'):
    load_scene = getattr(config, load_key)
    if load_scene is not None:
        return SingleSceneLoader(scene_component, load_scene)
    
    if train:
        if config.train_dataset is not None:
            return DatasetLoader(
                scene_component,
                config.train_dataset,
                config.train_split,
                subset=config.train_subset,
                repeat=config.train_repeat,
                shuffle=config.train_shuffle,
            )
    else:
        if config.eval_dataset is not None:
            return DatasetLoader(
                scene_component,
                config.eval_dataset,
                config.eval_split,
                subset=config.eval_subset,
                repeat=config.eval_repeat,
                shuffle=config.eval_shuffle,
            )
    
    return ClearScene(scene_component)
