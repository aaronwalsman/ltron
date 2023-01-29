from collections import OrderedDict

from steadfast.hierarchy import hierarchy_getitem

from supermecha import (
    SuperMechaComponent,
    SuperMechaContainer,
    TimeStepComponent,
)

from ltron.dataset.info import get_dataset_info
from ltron.gym.components import (
    EmptySceneComponent,
    DatasetLoader,
    VisualInterfaceConfig,
    VisualInterface,
    ColorRenderComponent
)

class SelectConnectionPointConfig(VisualInterfaceConfig):
    max_time_steps = 20
    image_height = 256
    image_width = 256

class SelectConnectionPointReward(SuperMechaComponent):
    def __init__(self, scene_component, cursor_component):
        self.scene_component = scene_component
        self.cursor_component = cursor_component
    
    def step(self, action):
        instance, snap = self.cursor_component.click_snap
        if instance != 0:
            return None, 1., True, False, {}
        else:
            return None, 0., False, False, {}

class SelectConnectionPointEnv(SuperMechaContainer):
    
    def __init__(self,
        config,
        dataset_name,
        dataset_split,
        dataset_subset=None,
        dataset_repeat=1,
        dataset_shuffle=True,
        train=True,
    ):
        components = OrderedDict()
        dataset_info = get_dataset_info(dataset_name)
        
        # scene
        components['scene'] = EmptySceneComponent(
            dataset_info['shape_ids'],
            dataset_info['color_ids'],
            dataset_info['max_instances_per_scene'],
            dataset_info['max_edges_per_scene'],
            track_snaps=True,
            collision_checker=True,
        )
        
        # dataset loader
        components['loader'] = DatasetLoader(
            components['scene'],
            dataset_name,
            dataset_split,
            subset=dataset_subset,
            shuffle=dataset_shuffle,
            shuffle_buffer=1000,
            repeat=dataset_repeat,
        )
        
        # time step
        components['time'] = TimeStepComponent(1, observe_step=False)
        
        # visual interface
        components['interface'] = VisualInterface(
            config,
            components['scene'],
            dataset_info['max_instances_per_scene'],
        )
        
        # color render
        components['image'] = ColorRenderComponent(
            components['scene'],
            config.image_height,
            config.image_width,
            anti_alias=True,
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            observable=True,
        )
        
        # reward
        components['reward'] = SelectConnectionPointReward(
            components['scene'],
            components['interface'].components['cursor'],
        )
        
        super().__init__(components)
