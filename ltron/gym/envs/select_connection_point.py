from collections import OrderedDict

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
)

class SelectConnectionPointConfig(VisualInterfaceConfig):
    pass

class SelectConnectionPointReward(SuperMechaComponent):
    def __init__(self, scene_component, cursor_component):
        self.scene_component = scene_component
        self.cursor_component = cursor_component
    
    def step(self, action):
        instance, snap = cursor_component.get_selected_snap()
        if instance != 0:
            return None, 1., True, False, None
        else:
            return None, 0., False, False, None

class SelectConnectionPointEnv(SuperMechaContainer):
    def __init__(self,
        config,
        dataset_name,
        dataset_split,
        dataset_subset=None,
        dataset_repeat=1,
        dataset_shuffle=True,
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
            include_manipulation=True,
            include_floating_pane=False,
            include_removal=False,
        )
        
        # shape_prediction
        components['reward'] = SelectConnectionPointReward(
            components['scene'],
            components['interface'].components['cursor'].components['pick'],
        )
        
        super().__init__(components)
