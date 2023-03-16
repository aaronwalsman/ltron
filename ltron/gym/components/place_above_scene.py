from supermecha import SuperMechaComponent

class PlaceAboveScene(SuperMechaComponent):
    def __init__(self,
        scene_component,
        offset=(0,48,0),
    ):
        self.scene_component = scene_component
        self.offset = offset
    
    def reset(self, seed=None, options=None):
        super().reset(seed)
        scene = self.scene_component.brick_scene
        instances = list(scene.instances.values())
        if len(instances):
            instance = self.np_random.choice(instances)
            scene.place_above_scene(
                [instance], offset=self.offset)
        
        return None, {}
