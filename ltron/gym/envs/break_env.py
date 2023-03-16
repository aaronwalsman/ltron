from collections import OrderedDict

from supermecha import (
    SuperMechaComponent,
    SuperMechaContainer,
    TimeStepComponent,
)

from ltron.dataset.info import get_dataset_info
from ltron.gym.components import (
    EmptySceneComponent,
    make_loader,
    VisualInterfaceConfig,
    make_visual_interface,
    ColorRenderComponent,
    AssemblyComponent,
)

class BreakEnvConfig(VisualInterfaceConfig):
    load_scene = None
    train_dataset_name = None
    train_split = None
    eval_dataset_name = None
    eval_split = None
    dataset_subset = None
    dataset_repeat = 1
    dataset_shuffle = True
    
    max_time_steps = 20
    
    image_height = 256
    image_width = 256
    render_mode = 'egl'
    
    shape_class_labels = None
    color_class_labels = None

class BreakEnv(SuperMechaContainer):
    def __init__(self,
        config,
        train=True,
    ):
        components = OrderedDict()
        
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
            renderable=True,
            render_args=render_args,
            track_snaps=True,
            collision_checker=True,
        )
        
        # loader
        components['loader'] = make_loader(
            config, components['scene'], train=train)
        
        # time step
        components['time'] = TimeStepComponent(
            config.max_time_steps, observe_step=True)
        
        # visual interface
        interface_components = make_visual_interface(
            config,
            components['scene'],
            train=train,
        )
        components.update(interface_components)
        
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
            shape_class_labels=config.shape_class_labels,
            color_class_labels=config.color_class_labels,
            #max_instances=dataset_info['max_instances_per_scene'],
            #max_edges=dataset_info['max_edges_per_scene'],
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            observable=True,
        )
        
        super().__init__(components)
