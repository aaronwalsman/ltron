from gym.spaces import Dict, Discrete

from ltron.hierarchy import hierarchy_branch
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.bricks.brick_scene import BrickScene

class SceneComponent(LtronGymComponent):
    def __init__(self,
        dataset_component=None,
        path_location=None,
        initial_scene_path=None,
        renderable=True,
        render_args=None,
        track_snaps=False,
        collision_checker=False,
    ):
        
        self.dataset_component = dataset_component
        self.path_location = path_location
        self.initial_scene_path = initial_scene_path
        self.current_scene_path = None
        
        if render_args is None:
            render_args={'opengl_mode':'egl', 'load_scene':'grey_cube'}
            
        self.brick_scene = BrickScene(
            renderable=renderable,
            render_args=render_args,
            track_snaps = track_snaps,
            collision_checker = collision_checker,
        )
        
        if self.initial_scene_path is not None:
            self.brick_scene.import_ldraw(initial_scene_path)
        
        self.observation_space = Dict({'valid_scene_loaded':Discrete(2)})
    
    def compute_observation(self):
        return {'valid_scene_loaded' : int(self.current_scene_path is not None)}
    
    def reset(self):
        self.brick_scene.clear_instances()
        if self.dataset_component is not None:
            #self.current_scene_path = magic_lookup(
            #    self.dataset_component, self.path_location)
            self.current_scene_path = hierarchy_branch(
                self.dataset_component.dataset_item, self.path_location)
        elif self.initial_scene_path is not None:
            self.current_scene_path = self.initial_scene_path
        '''
        if self.path_component is not None:
            self.current_scene_path = self.path_component.scene_path
        elif self.initial_scene_path is not None:
            self.current_scene_path = self.initial_scene_path
        '''
        if self.current_scene_path is not None:
            self.brick_scene.import_ldraw(self.current_scene_path)
        
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None
    
    def set_state(self, state):
        self.brick_scene.clear_assets()
        self.brick_scene.clear_instances()
        if self.dataset_component is not None:
            self.brick_scene.import_ldraw(self.dataset_component.scene_path)
        elif self.initial_scene_path is not None:
            self.brick_scene.import_ldraw(self.initial_scene_path)
