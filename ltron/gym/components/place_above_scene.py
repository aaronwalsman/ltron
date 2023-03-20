from supermecha import SuperMechaComponent

from ltron.geometry.collision import build_collision_map
from ltron.geometry.utils import orthogonal_orientations

class PlaceAboveScene(SuperMechaComponent):
    def __init__(self,
        scene_component,
        offset=(0,48,0),
        randomize_orientation=False,
    ):
        self.scene_component = scene_component
        self.offset = offset
        self.randomize_orientation = randomize_orientation
    
    def reset(self, seed=None, options=None):
        super().reset(seed)
        scene = self.scene_component.brick_scene
        collision_map = build_collision_map(scene)
        removable_instances = [
            i for i, groups in collision_map.items()
            if all([len(v) == 0 for v in groups.values()])
        ]
        #instances = list(scene.instances.values())
        if len(removable_instances):
            instance = scene.instances[
                self.np_random.choice(removable_instances)]
            scene.place_above_scene(
                [instance], offset=self.offset)
            if self.randomize_orientation:
                instance_transform = instance.transform.copy()
                orientations = orthogonal_orientations()
                orientation = self.np_random.choice(orientations)
                instance_transform[:3,:3] = orientation[:3,:3]
                scene.move_instance(instance, instance_transform)
        
        return None, {}
