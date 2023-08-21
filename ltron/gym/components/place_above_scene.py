import numpy

from supermecha import SuperMechaComponent

from ltron.geometry.collision import build_collision_map
from ltron.geometry.utils import orthogonal_orientations

class PlaceAboveScene(SuperMechaComponent):
    def __init__(self,
        scene_component,
        offset=(0,48,0),
        randomize_orientation=False,
        randomize_orientation_mode=24,
        selection_mode='random',
    ):
        self.scene_component = scene_component
        self.offset = offset
        self.randomize_orientation = randomize_orientation
        self.randomize_orientation_mode = randomize_orientation_mode
        self.selection_mode = selection_mode
    
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
            if self.selection_mode == 'random':
                instance = self.np_random.choice(removable_instances)
                instance = scene.instances[instance]
            elif self.selection_mode == 'highest':
                #removable_instances = [
                #    scene.instances[i] for i in removable_instances]
                heights = [
                    (scene.instances[i].transform[1,3], i)
                    for i in removable_instances
                ]
                instance = max(heights)[1]
                instance = scene.instances[instance]
            else:
                raise ValueError(
                    'Unknown selection_mode: %s'%self.selection_mode)
            #connections = scene.get_instance_snap_connections(instance)
            #snap, _ = self.np_random.choice(connections)
            #snap_inv = numpy.linalg.inv(snap.transform)
            scene.place_above_scene(
                [instance], offset=self.offset)
            if self.randomize_orientation:
                instance_transform = instance.transform.copy()
                if self.randomize_orientation_mode == 24:
                    orientations = orthogonal_orientations()
                else:
                    orientations = [
                        #scene.upright @
                        #snap_inv,
                        #numpy.eye(4),
                        scene.upright,
                        scene.upright @
                        numpy.array([
                            [ 0, 0, 1, 0],
                            [ 0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [ 0, 0, 0, 1]]),
                        scene.upright @
                        numpy.array([
                            [-1, 0, 0, 0],
                            [ 0, 1, 0, 0],
                            [ 0, 0,-1, 0],
                            [ 0, 0, 0, 1]]),
                        scene.upright @
                        numpy.array([
                            [ 0, 0,-1, 0],
                            [ 0, 1, 0, 0],
                            [ 1, 0, 0, 0],
                            [ 0, 0, 0, 1]]),
                    ]
                orientation = self.np_random.choice(orientations)
                instance_transform[:3,:3] = orientation[:3,:3]
                scene.move_instance(instance, instance_transform)
        
        return None, {}
