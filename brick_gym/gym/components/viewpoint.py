import math
import random

import numpy

import gym.spaces as spaces

import renderpy.camera as camera

from brick_gym.gym.components.brick_env_component import BrickEnvComponent

class ControlledAzimuthalViewpointComponent(BrickEnvComponent):
    def __init__(self,
            scene_component,
            azimuth_steps,
            elevation_range,
            elevation_steps,
            distance_range,
            distance_steps,
            field_of_view = math.radians(60.),
            aspect_ratio = 1.,
            near_clip = 1.,
            far_clip = 5000.,
            start_position = 'uniform'):
        
        self.scene_component = scene_component
        self.azimuth_steps = azimuth_steps
        self.elevation_range = elevation_range
        self.elevation_steps = elevation_steps
        self.distance_range = distance_range
        self.distance_steps = distance_steps
        self.field_of_view = field_of_view
        self.aspect_ratio = aspect_ratio
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.start_position = start_position
        
        self.observation_space = spaces.Dict({
                'azimuth' : spaces.Discrete(azimuth_steps),
                'elevation' : spaces.Discrete(elevation_steps),
                'distance' : spaces.Discrete(distance_steps)
        })
        self.action_space = spaces.Discrete(7)
        self.num_locations = azimuth_steps * elevation_steps * distance_steps
        self.location = None
        
        self.azimuth_spacing = math.pi * 2 / azimuth_steps
        self.elevation_spacing = (
                elevation_range[1] - elevation_range[0]) / (elevation_steps-1)
        self.distance_spacing = (
                distance_range[1] - distance_range[0]) / (distance_steps-1)
    
    def compute_observation(self):
        return {'azimuth' : self.position[0],
                'elevation' : self.position[1],
                'distance' : self.position[2]}
                
    
    def reset(self):
        if self.start_position == 'uniform':
            self.position = [
                    random.randint(0, self.azimuth_steps-1),
                    random.randint(0, self.elevation_steps-1),
                    random.randint(0, self.distance_steps-1)]
        else:
            self.position = list(start_position)
        self.set_camera()
        
        return self.compute_observation()
    
    def step(self, action):
        if action == 0:
            pass
        elif action == 1:
            self.position[0] -= 1
            self.position[0] = self.position[0] % self.azimuth_steps
        elif action == 2:
            self.position[0] += 1
            self.position[0] = self.position[0] % self.azimuth_steps
        elif action == 3:
            self.position[1] -= 1
            self.position[1] = max(0, self.position[1])
        elif action == 4:
            self.position[1] += 1
            self.position[1] = min(self.elevation_steps-1, self.position[1])
        elif action == 5:
            self.position[2] -= 1
            self.position[2] = max(0, self.position[2])
        elif action == 6:
            self.position[2] += 1
            self.position[2] = min(self.distance_steps-1, self.position[2])
        
        self.set_camera()
        
        #tmp_reward = self.position[1] + self.position[2]
        
        return self.compute_observation(), 0., False, None
    
    def set_camera(self):
        scene = self.scene_component.brick_scene
        azimuth = self.position[0] * self.azimuth_spacing
        elevation = (self.position[1] * self.elevation_spacing +
                self.elevation_range[0])
        field_of_view = self.field_of_view
        distance = (self.position[2] * self.distance_spacing +
                self.distance_range[0])
        
        # projection
        self.projection = camera.projection_matrix(
                self.field_of_view,
                self.aspect_ratio,
                self.near_clip,
                self.far_clip)
        scene.set_projection(self.projection)
        
        # pose
        bbox = scene.get_instance_center_bbox()
        bbox_min, bbox_max = bbox
        bbox_range = numpy.array(bbox_max) - numpy.array(bbox_min)
        center = bbox_min + bbox_range * 0.5
        camera_pose = camera.azimuthal_pose_to_matrix(
                [azimuth, elevation, 0, distance, 0.0, 0.0],
                center = center)
        scene.set_camera_pose(camera_pose)

class RandomizedAzimuthalViewpointComponent(BrickEnvComponent):
    def __init__(self,
            scene_component,
            azimuth = (0, math.pi*2),
            elevation = (math.radians(-15), math.radians(-45)),
            tilt = (math.radians(-45.), math.radians(45.)),
            field_of_view = (math.radians(60.), math.radians(60.)),
            distance = (0.8, 1.2),
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
        self.distance = distance
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
        distance_scale = random.uniform(*self.distance)
        
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
        distance = distance_scale * camera.framing_distance_for_bbox(
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
