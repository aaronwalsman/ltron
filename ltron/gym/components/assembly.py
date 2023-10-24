import numpy

from supermecha import SensorComponent

from ltron.constants import (
    SHAPE_CLASS_LABELS,
    COLOR_CLASS_LABELS,
    MAX_INSTANCES_PER_SCENE,
    MAX_EDGES_PER_SCENE,
)
from ltron.bricks.brick_scene import TooManyInstancesError, make_empty_assembly
from ltron.gym.spaces import AssemblySpace
from ltron.geometry.collision import build_collision_map

class AssemblyComponent(SensorComponent):
    def __init__(self,
        scene_component,
        shape_class_labels=None,
        color_class_labels=None,
        max_instances=None,
        max_edges=None,
        compute_collision_map=False,
        update_on_init=False,
        update_on_reset=False,
        update_on_step=False,
        observable=True,
        truncate_if_unchanged=False,
    ):
        super().__init__(
            update_on_init=update_on_init,
            update_on_reset=update_on_reset,
            update_on_step=update_on_step,
            observable=observable,
            truncate_if_unchanged=truncate_if_unchanged,
        )
        
        self.scene_component = scene_component
        self.shape_class_labels = shape_class_labels
        self.color_class_labels = color_class_labels
        if max_instances is None:
            max_instances = MAX_INSTANCES_PER_SCENE
        self.max_instances = max_instances
        if max_edges is None:
            max_edges = MAX_EDGES_PER_SCENE
        self.max_edges = max_edges
        
        if self.observable:
            self.observation_space = AssemblySpace(
                self.max_instances,
                self.max_edges,
            )
        
        self.compute_collision_map = compute_collision_map
    
    def step(self, action):
        #u = False
        #if len(self.scene_component.brick_scene.instances) > self.max_instances:
        #    o = EMPTY_ASSEMBLY
        #    return o, 0., False, True, {}
        #else:
        #    return super().step(action)
        o,r,t,u,i = super().step(action)
        scene = self.scene_component.brick_scene
        if len(scene.instances) == self.max_instances:
            u = True
        return o,r,t,u,i
    
    def compute_observation(self):
        assembly = self.scene_component.brick_scene.get_assembly(
            shape_class_labels=self.shape_class_labels,
            color_class_labels=self.color_class_labels,
            max_instances=self.max_instances,
            max_edges=self.max_edges,
        )
        
        if self.compute_collision_map:
            self.collision_map = build_collision_map(
                self.scene_component.brick_scene,
            )
        
        return assembly, {}
