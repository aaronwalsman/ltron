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
    ColorRenderComponent,
    AssemblyComponent,
)

class BreakEnvConfig(VisualInterfaceConfig):
    max_time_steps = 20
    image_height = 256
    image_width = 256
    render_mode = 'egl'

class BreakEnv(SuperMechaContainer):
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
        if config.render_mode == 'egl':
            render_args = None
        elif config.render_mode == 'glut':
            render_args = {
                'opengl_mode' : 'glut',
                'window_width' : config.image_width,
                'window_height' : config.image_height,
                'load_scene' : 'front_light',
            }
        components['scene'] = EmptySceneComponent(
            dataset_info['shape_ids'],
            dataset_info['color_ids'],
            dataset_info['max_instances_per_scene'],
            dataset_info['max_edges_per_scene'],
            renderable=True,
            render_args=render_args,
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
            train=train,
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
        
        components['assembly'] = AssemblyComponent(
            components['scene'],
            dataset_info['shape_ids'],
            dataset_info['color_ids'],
            dataset_info['max_instances_per_scene'],
            dataset_info['max_edges_per_scene'],
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            observable=True,
        )
        
        super().__init__(components)
    
    #def step(self, *args, **kwargs):
    #    o,r,t,u,i = super().step(*args, **kwargs)
    #    print('reward: %.04f'%r)
    #    return o,r,t,u,i
