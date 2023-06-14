from supermecha import SuperMechaComponent

from ltron.gym.spaces import AssemblySpace
from ltron.hierarchy import hierarchy_branch
from ltron.bricks.brick_scene import BrickScene

class EmptySceneComponent(SuperMechaComponent):
    def __init__(self,
        renderable=True,
        render_args=None,
        track_snaps=False,
        collision_checker=False,
        clear_frequency='reset',
    ):
        self.clear_frequency = clear_frequency
        
        if render_args is None:
            render_args = {'opengl_mode':'egl', 'load_scene':'front_light'}
        
        self.brick_scene = BrickScene(
            renderable=renderable,
            render_args=render_args,
            track_snaps=track_snaps,
            collision_checker=collision_checker,
        )
    
    def clear_scene(self):
        self.brick_scene.clear_instances()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self.clear_frequency in ('step', 'reset'):
            self.clear_scene()
        return None, {}
    
    def step(self, action):
        if self.clear_frequency in ('step',):
            self.clear_scene()
        return None, 0., False, False, {}
    
    def set_state(self, state):
        self.brick_scene.clear_instances()
        self.brick_scene.set_assembly(state)
        
        return None, {}
    
    def get_state(self):
        self.brick_scene_get_assembly(state)
        
        return state


class SingleSceneComponent(EmptySceneComponent):
    def __init__(self,
        initial_scene_path,
        *args,
        **kwargs
    ):
        super(SingleSceneComponent, self).__init__(*args, **kwargs)
        self.current_scene_path = initial_scene_path
