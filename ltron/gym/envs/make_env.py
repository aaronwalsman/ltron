from collections import OrderedDict

from supermecha import (
    SuperMechaComponent,
    SuperMechaContainer,
    TimeStepComponent,
)

from ltron.dataset.info import get_dataset_info
from ltron.gym.components import (
    EmptySceneComponent,
    LoaderConfig,
    make_loader,
    ClearScene,
    VisualInterfaceConfig,
    make_visual_interface,
    ColorRenderComponent,
    AssemblyComponent,
    BuildScore
)

class MakeEnvConfig(VisualInterfaceConfig, LoaderConfig):
    load_scene = None
    load_start_scene = None
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

class MakeEnv(SuperMechaContainer):
    def __init__(self,
        config,
        train=False,
    ):
        components = OrderedDict()
        #dataset_info = get_dataset_info(config.dataset_name)
        
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
        components['target_assembly'] = AssemblyComponent(
            components['scene'],
            #dataset_info['max_instances_per_scene'],
            #dataset_info['max_edges_per_scene'],
            shape_class_labels=config.shape_class_labels,
            color_class_labels=config.color_class_labels,
            update_on_init=False,
            update_on_reset=True,
            update_on_step=False,
            observable=True,
        )
        if config.load_start_scene is None:
            components['clear_scene'] = ClearScene(components['scene'])
        else:
            components['start_loader'] = make_loader(
                config,
                components['scene'],
                train=train,
                load_key='load_start_scene',
            )
        
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
            #max_instances = dataset_info['max_instances_per_scene'],
            #max_edges = dataset_info['max_edges_per_scene'],
            shape_class_labels=config.shape_class_labels,
            color_class_labels=config.color_class_labels,
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            observable=True,
        )
        components['score'] = BuildScore(
            components['target_assembly'],
            components['assembly'],
        )
        
        super().__init__(components)
