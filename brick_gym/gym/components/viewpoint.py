import math
import random

import numpy

import renderpy.camera as camera

from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class RandomizedAzimuthalViewpointComponent(BrickEnvComponent):
    def __init__(self,
            scene_component,
            azimuth = (0, math.pi*2),
            elevation = (math.radians(-15), math.radians(-45)),
            tilt = (math.radians(-45.), math.radians(45.)),
            field_of_view = (math.radians(60.), math.radians(60.)),
            aspect_ratio = 1.,
            near_clip = 1.,
            far_clip = 5000.,
            bbox_distance_scale = 3.,
            randomize_frequency = 'reset'):
        
        self.scene_component = scene_component
        self.scene_component.brick_scene.make_renderable()
        self.azimuth = azimuth
        self.elevation = elevation
        self.tilt = tilt
        self.field_of_view = field_of_view
        self.aspect_ratio = aspect_ratio
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.bbox_distance_scale = bbox_distance_scale
        self.randomize_frequency = randomize_frequency
        
        self.set_camera()
    
    def set_camera(self):
        # projection
        scene = self.scene_component.brick_scene
        azimuth = random.uniform(*self.azimuth)
        elevation = random.uniform(*self.elevation)
        tilt = random.uniform(*self.tilt)
        field_of_view = random.uniform(*self.field_of_view)
        
        self.projection = camera.projection_matrix(
                field_of_view,
                self.aspect_ratio,
                self.near_clip,
                self.far_clip)
        scene.set_projection(self.projection)
        
        # pose
        bbox = scene.get_instance_center_bbox()
        bbox_min, bbox_max = bbox
        bbox_range = numpy.array(bbox_max) - numpy.array(bbox_min)
        center = bbox_min + bbox_range * 0.5
        distance = camera.framing_distance_for_bbox(
                bbox, self.projection, self.bbox_distance_scale)
        camera_pose = camera.azimuthal_pose_to_matrix(
                [azimuth, elevation, tilt, distance, 0.0, 0.0],
                center = center)
        scene.set_camera_pose(camera_pose)
    
    def reset(self):
        self.set_camera()
    
    def step(self, action):
        if self.randomize_frequency == 'step':
            self.set_camera()
        return None, 0., False, None
    
    def set_state(self, state):
        self.set_camera()

class FixedAzimuthalViewpointComponent(RandomizedAzimuthalViewpointComponent):
    def __init__(self,
            scene_component,
            azimuth,
            elevation,
            tilt = 0.,
            field_of_view = math.radians(60.),
            *args, **kwargs):
        
        super(FixedAzimuthalViewpointComponent, self).__init__(
                scene_component,
                (azimuth, azimuth),
                (elevation, elevation),
                (tilt, tilt),
                (field_of_view, field_of_view),
                *args, **kwargs)
