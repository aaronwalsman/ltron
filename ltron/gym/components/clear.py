#from ltron.gym.components.sensor_component import SensorComponent
from supermecha.gym.components.sensor_component import SensorComponent

class ClearScene(SensorComponent):
    def __init__(self,
        scene_component,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            observable=False,
        )
        self.scene_component = scene_component
    
    def compute_observation(self):
        scene = self.scene_component.brick_scene
        scene.clear_instances()
        
        return None, None
