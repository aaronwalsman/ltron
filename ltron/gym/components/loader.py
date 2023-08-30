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
    def __init__(self, scene_component, file_paths, center_assembly=False):
        self.scene_component = scene_component
        self.file_paths = file_paths.split(',')
        self.center_assembly = center_assembly
        self.loads = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed)
        scene = self.scene_component.brick_scene
        self.scene_component.clear_scene()
        #file_path = self.file_paths[self.loads%len(self.file_paths)]
        file_path = self.np_random.choice(self.file_paths)
        #print(file_path)
        scene.import_ldraw(file_path)
        
        if self.center_assembly:
            box_min, box_max = scene.get_bbox()
            center = (box_max + box_min)*0.5
            x_center = round(center[0] / 20.) * 20.
            y_center = round(center[1] / 8.) * 8.
            z_center = round(center[2] / 20.) * 20.
            for instance in scene.instances.values():
                t = instance.transform.copy()
                t[0,3] -= x_center
                t[1,3] -= y_center
                t[2,3] -= z_center
                scene.move_instance(instance, t)
        
        self.loads += 1
        
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
        shuffle_buffer=10000,
        repeat=False,
        center_assembly=False,
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
        self.center_assembly = center_assembly
        
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
                self.file_name = datapoint['__key__']
                text = datapoint['mpd']
                scene = self.scene_component.brick_scene
                scene.import_text(datapoint['__key__'] + '.mpd', text)
                if self.center_assembly:
                    box_min, box_max = scene.get_bbox()
                    center = (box_min + box_max)*0.5
                    x_center = round(center[0] / 20.) * 20.
                    y_center = round(center[1] / 8.) * 8.
                    z_center = round(center[2] / 20.) * 20.
                    for instance in scene.instances.values():
                        t = instance.transform.copy()
                        t[0,3] -= x_center
                        t[1,3] -= y_center
                        t[2,3] -= z_center
                        scene.move_instance(instance, t)
        
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
    eval_shuffle = True
    center_assembly = False

def make_loader(config, scene_component, train=False, load_key='load_scene'):
    load_scene = getattr(config, load_key)
    if load_scene is not None:
        return SingleSceneLoader(
            scene_component, load_scene, center_assembly=config.center_assembly)
    
    if train:
        if config.train_dataset is not None:
            return DatasetLoader(
                scene_component,
                config.train_dataset,
                config.train_split,
                subset=config.train_subset,
                repeat=config.train_repeat,
                shuffle=config.train_shuffle,
                center_assembly=config.center_assembly,
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
                center_assembly=config.center_assembly,
            )
    
    return ClearScene(scene_component)
