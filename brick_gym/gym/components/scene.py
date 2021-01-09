from brick_gym.gym.components.brick_env_component import BrickEnvComponent
from brick_gym.bricks.brick_scene import BrickScene

class SceneComponent(BrickEnvComponent):
    def __init__(self,
            path_component=None,
            initial_scene_path=None,
            default_image_light='grey_cube'):
        
        self.path_component = path_component
        self.initial_scene_path = initial_scene_path
        
        self.brick_scene = BrickScene(default_image_light = default_image_light)
        
        if self.initial_scene_path is not None:
            self.brick_scene.import_ldraw(initial_scene_path)
    
    def reset(self):
        if self.path_component is not None:
            scene_path = self.path_component.scene_path
            self.brick_scene.clear_instances()
            self.brick_scene.import_ldraw(scene_path)
    
    def set_state(self, state):
        self.brick_scene.clear_assets()
        self.brick_scene.clear_instances()
        if self.path_component is not None:
            self.brick_scene.import_ldraw(self.path_component.scene_path)
        elif self.initial_scene_path is not None:
            self.brick_scene.import_ldraw(self.initial_scene_path)
