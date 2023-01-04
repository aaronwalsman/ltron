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

class BreakEnvConfig(VisualInterfaceConfig):
    max_time_steps = 20

class BreakEnvRenderBasedReward(SuperMechaComponent):
    def __init__(self,
        scene_component,
        instance_render_component,
        target_assembly_component,
    ):
        self.scene_component = scene_component
        self.instance_render_component = instance_render_component
    
    def update_observed_instances(self):
        instance_map = self.instance_render_component.observe()
        visible_instances = set(numpy.unique(instance_map))
        self.observed_instances |= visible_instances
    
    def reset(self, seed, rng):
        super().reset(self, seed, rng)
        target_assembly = self.target_assembly_component.observe()
        target_instances = numpy.where(target_assembly['shape'])
        self.target_instances = set()
        self.observed_instances = set()
        self.update_observed_instances()
        
        return None, {}
    
    def step(self, action):
        initial_instances = len(self.observed_instances)
        self.update_observed_instances()
        final_instances = len(self.observed_instances)
        reward_scale = 1. / len(self.target_instances)
        reward = (final_instances - initial_instances) * reward_scale
        
        return None, reward, False, False, {}

class BreakEnv(SuperMechaContainer):
    
    allow_none_info = False
    
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
        components['time'] = TimeStepComponent(
            config.max_time_steps, observe_step=False)
        
        # visual interface
        components['interface'] = VisualInterface(
            config,
            components['scene'],
            dataset_info['max_instances_per_scene'],
            include_manipulation=True,
            include_floating_pane=False,
            include_brick_removal=True,
        )
        
        ## reward
        # this is computed as a wrapper
        #components['reward'] = BreakEnvReward(
        #    components['scene'],
        #)
        
        super().__init__(components)
