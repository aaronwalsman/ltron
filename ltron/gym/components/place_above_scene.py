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
        #number_to_remove=1,
    ):
        self.scene_component = scene_component
        self.offset = offset
        self.randomize_orientation = randomize_orientation
        self.randomize_orientation_mode = randomize_orientation_mode
        self.selection_mode = selection_mode
        #self.number_to_remove = number_to_remove
    
    def reset(self, seed=None, options=None):
        super().reset(seed)
        scene = self.scene_component.brick_scene
        #collision_map = build_collision_map(scene)
        ##removable_instances = [
        ##    i for i, groups in collision_map.items()
        ##    if all([len(v) == 0 for v in groups.values()])
        ##]
        #removable_instances = []
        #for i, groups in collision_map.items():
        #    for c in groups.values():
        #        if not len(c):
        #            removable_instances.append(i)
        #            break
        
        #if self.number_to_remove == 'uniform':
        #    number_to_remove = self.np_random.integers(
        #        low=1, high=len(scene.instances), size=1)[0]
        #else:
        #    number_to_remove = self.number_to_remove
        
        #for i in range(number_to_remove):
        #    # not using collision maps anymore
        #    if self.selection_mode == 'highest':
        #        removable_instances = [int(i) for i in scene.instances]
        #    else:
        #        removable_instances = [
        #            int(i) for i in scene.instances
        #            if scene.instance_captive(i)
        #        ]
        #    if i < number_to_remove-1:
        #        if self.selection_mode == 'random':
        #            instance = self.np_random.choice(removable_instances)
        #            instance = scene.instances[instance]
        #        elif self.selection_mode == 'highest':
        #            heights = [
        #                (scene.instances[i].transform[1,3], i)
        #                for i in removable_instances
        #            ]
        #            instance = max(heights)[1]
        #            instance = scene.instances[instance]
        #        scene.remove_instance(instance)
        removable_instances = [
            int(i) for i in scene.instances
            if scene.instance_captive(i)
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
