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
    BuildScore,
    BreakAndMakePhaseSwitchComponent,
    PhaseScoreComponent,
)

class BreakAndMakeEnvConfig(VisualInterfaceConfig, LoaderConfig):
    image_height = 256
    image_width = 256
    render_mode = 'egl'
    compute_collision_map = False

    max_instances = None

class BreakAndMakeEnv(SuperMechaContainer):
    def __init__(self,
        config,
        train=False,
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
        
        # target assembly
        components['target_assembly'] = AssemblyComponent(
            components['scene'],
            update_on_init=False,
            update_on_reset=True,
            update_on_step=False,
            observable=True,
        )
        
        # visual interface
        config.include_done = False
        config.include_phase = True
        interface_components = make_visual_interface(
            config,
            components['scene'],
            train=train,
        )
        #interface_components['phase'] = BreakAndMakePhaseSwitchComponent(
        #    components['scene'],
        #)
        
        # separate out the render and nonrender components from the interface
        render_components = {
            k:v for k,v in interface_components.items()
            if 'render' in k
        }
        nonrender_components = {
            k:v for k,v in interface_components.items()
            if 'render' not in k
        }
        components.update(nonrender_components)
        
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
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            observable=True,
            compute_collision_map=config.compute_collision_map,
        )
        components.update(render_components)
        
        # score
        score_component = BuildScore(
            components['target_assembly'],
            components['assembly'],
        )
        components['score'] = PhaseScoreComponent(
            #components['phase'],
            components['action_primitives'].components['phase'],
            score_component,
        )
        
        super().__init__(components)
