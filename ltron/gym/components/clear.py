from ltron.gym.components.sensor_component import SensorComponent

class ClearScene(SensorComponent):
    def __init__(self,
        scene_components,
        update_frequency='reset',
    ):
        super().__init__(
            update_frequency=update_frequency,
            observable=False,
        )
        self.scene_components = scene_components
        self.observation = None
    
    def update_observation(self):
        for name, component in self.scene_components.items():
            scene = component.brick_scene
            scene.clear_instances()
        
        self.observation = None
