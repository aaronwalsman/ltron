import math

import numpy

import renderpy.camera as camera

from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class FixedAzimuthalViewpointComponent(BrickEnvComponent):
    def __init__(self,
            azimuth = 0,
            elevation = 0,
            field_of_view = math.radians(90.),
            aspect_ratio = 1.,
            near_clip = 1.,
            far_clip = 5000.,
            bbox_distance_scale = 3.,
            scene_component):
        
        self.azimuth = azimuth
        self.elevation = elevation
        self.field_of_view = field_of_view
        self.aspect_ratio = aspect_ratio
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.bbox_distance_scale = bbox_distance_scale
        self.scene_component = scene_component
        
        self.set_camera()
    
    def set_camera(self):
        # projection
        scene = self.scene_component.brick_scene
        self.projection = camera.projection_matrix(
                self.fov, self.aspect_ratio, self.near_clip, self.far_clip)
        scene.set_projection(self.projection)
        
        # pose
        bbox = scene.get_instance_center_bbox()
        bbox_min, bbox_max = bbox
        bbox_range = numpy.array(bbox_max) - numpy.array(bbox_min)
        center = bbox_min + bbox_range * 0.5
        distance = camera.framing_distance_for_bbox(
                bbox, self.projection, self.bbox_distance_scale)
        camera_pose = camera.azimuthal_pose_to_matrix(
                [self.azimuth, self.elevation, 0.0, distance, 0.0, 0.0],
                center = center)
        scene.set_camera_pose(camera_pose)
    
    def reset(self):
        self.set_camera()
    
    def set_state(self, state):
        self.set_camera()
