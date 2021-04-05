from brick_gym.gym.components.brick_env_component import BrickEnvComponent
from brick_gym.bricks.brick_scene import BrickScene
from gym.spaces import Dict, Discrete

class SceneComponent(BrickEnvComponent):
    def __init__(self,
            path_component=None,
            initial_scene_path=None,
            renderable=True,
            default_image_light='grey_cube'):
        
        self.path_component = path_component
        self.initial_scene_path = initial_scene_path
        self.current_scene_path = None
        
        self.brick_scene = BrickScene(
                opengl_mode = 'egl',
                renderable=renderable,
                default_image_light = default_image_light)
        
        if self.initial_scene_path is not None:
            self.brick_scene.import_ldraw(initial_scene_path)
        
        self.observation_space = Dict({'valid_scene_loaded':Discrete(2)})
    
    def compute_observation(self):
        return {'valid_scene_loaded' : int(self.current_scene_path is not None)}
    
    def reset(self):
        self.brick_scene.clear_instances()
        if self.path_component is not None:
            self.current_scene_path = self.path_component.scene_path
        elif self.initial_scene_path is not None:
            self.current_scene_path = self.initial_scene_path
        
        if self.current_scene_path is not None:
            self.brick_scene.import_ldraw(self.current_scene_path)
        
        return self.compute_observation()
    
    def step(self, action):
        return self.compute_observation(), 0., False, None
    
    def set_state(self, state):
        self.brick_scene.clear_assets()
        self.brick_scene.clear_instances()
        if self.path_component is not None:
            self.brick_scene.import_ldraw(self.path_component.scene_path)
        elif self.initial_scene_path is not None:
            self.brick_scene.import_ldraw(self.initial_scene_path)
