from ltron.bricks.brick_scene import TooManyInstancesError, make_empty_assembly
from ltron.gym.spaces import AssemblySpace
from ltron.gym.components.sensor_component import SensorComponent

class AssemblyComponent(SensorComponent):
    def __init__(self,
        scene_component,
        shape_ids,
        color_ids,
        max_instances,
        max_edges,
        update_frequency,
        observable,
    ):
        super().__init__(
            update_frequency=update_frequency,
            observable=observable,
        )
        
        self.scene_component = scene_component
        self.shape_ids = shape_ids
        self.color_ids = color_ids
        self.max_instances = max_instances
        self.max_edges = max_edges
        
        if self.observable:
            self.observation_space = AssemblySpace(
                self.shape_ids,
                self.color_ids,
                self.max_instances,
                self.max_edges,
            )
    
    def update_observation(self):
        self.observation = self.scene_component.brick_scene.get_assembly(
            self.shape_ids,
            self.color_ids,
            self.max_instances,
            self.max_edges,
        )

class DeduplicateAssemblyComponent(SensorComponent):
    def __init__(self,
        assembly_component,
        update_frequency='step',
        observable=True,
    ):
        super().__init__(
            update_frequency=update_frequency,
            observable=observable,
        )
        self.assembly_component = assembly_component
        self.max_instances = self.assembly_component.max_instances
        
        if observable:
            self.observation_space = MaskedAssemblySpace(self.max_instances)
    
    def reset(self):
        super().reset()
        self.previous_shape = numpy.zeros(
            self.max_instances+1, dtype=numpy.long)
        self.previous_color = numpy.zeros(
            self.max_instances+1, dtype=numpy.long)
        self.previous_pose = numpy.zeros(
            (self.max_instances+1, 4, 4), dtype=numpy.float)
    
    def update_observation(self):
        # get the new assembly
        new_assembly = self.assembly_component.observe()
        
        # compare it the shape, color and pose
        shape_match = self.previous_shape == new_assembly['shape']
        color_match = self.previous_color == new_assembly['color']
        pose_match = self.previous_pose == new_assembly['pose']
        pose_match = numpy.all(pose_match, axis=(1,2))
        
        # store a mask indicating the elements that have changed
        self.observation = ~(shape_match & color_match & pose_match)
