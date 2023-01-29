from math import pi, radians
from enum import Enum

import numpy

from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Box

import splendor.camera as camera

from supermecha import SuperMechaComponent

from ltron.constants import DEFAULT_WORLD_BBOX

inf = float('inf')

ViewpointActions = Enum('ViewpointActions', [
    'NO_OP',
    'AZIMUTH_POS', 
    'AZIMUTH_NEG',
    'ELEVATION_POS',
    'ELEVATION_NEG',
    'DISTANCE_POS',
    'DISTANCE_NEG',
    'X_POS',
    'X_NEG',
    'Y_POS',
    'Y_NEG',
], start=0)

class ViewpointComponent(SuperMechaComponent):
    def __init__(self,
        scene_component=None,
        azimuth_steps=16,
        elevation_steps=5,
        elevation_range=(radians(-60.), radians(60.)),
        distance_steps=3,
        distance_range=(150.,450.),
        reset_mode='random',
        center_reset_range=((0.,0.,0.),(0.,0.,0.)),
        world_bbox=DEFAULT_WORLD_BBOX,
        allow_translate=True,
        translate_step_size=40.,
        field_of_view=radians(60.),
        aspect_ratio=1.,
        near_clip=10.,
        far_clip=50000.,
        observable=True,
    ):
        
        self.scene_component = scene_component
        self.azimuth_steps = azimuth_steps
        self.azimuth_step_size = pi * 2 / azimuth_steps
        self.elevation_steps = elevation_steps
        self.elevation_range = elevation_range
        self.elevation_step_size = (
            elevation_range[1] - elevation_range[0]) / (elevation_steps-1)
        self.distance_steps = distance_steps
        self.distance_range = distance_range
        self.distance_step_size = (
            distance_range[1] - distance_range[0]) / (distance_steps-1)
        self.reset_mode = reset_mode
        self.center_reset_range = center_reset_range
        self.world_bbox = world_bbox
        self.translate_step_size = translate_step_size
        self.field_of_view = field_of_view
        self.aspect_ratio = aspect_ratio
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.observable = observable
        
        # ensure the center starting range is within the center bounds
        assert all(
            [r >= b for r,b in zip(center_reset_range[0], world_bbox[0])])
        assert all(
            [r <= b for r,b in zip(center_reset_range[1], world_bbox[1])])
        
        # make action space
        if allow_translate:
            self.action_space = Discrete(len(ViewpointActions))
        else:
            self.action_space = Discrete(len(ViewpointActions)-4)
        
        # make observation space
        if self.observable:
            observation_space = {}
            if self.azimuth_steps > 1:
                observation_space['azimuth'] = Discrete(self.azimuth_steps)
            if self.elevation_steps > 1:
                observation_space['elevation'] = Discrete(
                    self.elevation_steps)
            if self.distance_steps > 1:
                observation_space['distance'] = Discrete(
                    self.distance_steps)
            if self.translate_step_size:
                observation_space['center'] = Box(
                    numpy.array(self.world_bbox[0], dtype=numpy.float32),
                    numpy.array(self.world_bbox[1], dtype=numpy.float32),
                    dtype=numpy.float32,
                )
            
            if len(observation_space):
                self.observation_space = Dict(observation_space)
    
    def compute_observation(self):
        if self.observable:
            self.observation = {}
            self.observation['azimuth'] = self.azimuth
            self.observation['elevation'] = self.elevation
            self.observation['distance'] = self.distance
            self.observation['center'] = numpy.array(
                [self.x, self.y, self.z], dtype=numpy.float32)
            
            return self.observation, {}
    
    def reset(self, seed=None, rng=None, options=None):
        super().reset(seed=seed, rng=rng, options=None)
        if self.reset_mode == 'random':
            self.azimuth = self.np_random.integers(0, self.azimuth_steps)
            self.elevation = self.np_random.integers(0, self.elevation_steps)
            self.distance = self.np_random.integers(0, self.distance_steps)
            self.x, self.y, self.z = [
                self.np_random.random() *
                (self.center_reset_range[1][i]-self.center_reset_range[0][i]) +
                self.center_reset_range[0][i]
                for i in range(3)
            ]
        
        else:
            (self.azimuth,
             self.elevation,
             self.distance,
             self.x,
             self.y,
             self.z) = self.reset_mode
        
        self.set_camera()
        self.compute_observation()
        return self.observation, {}
    
    def step(self, action):
        #if 'azimuth' in action:
        offset_vector = None
        viewpoint_action = ViewpointActions(value=action)
        if viewpoint_action == ViewpointActions.AZIMUTH_POS:
            self.azimuth += 1
        elif viewpoint_action == ViewpointActions.AZIMUTH_NEG:
            self.azimuth -= 1
        elif viewpoint_action == ViewpointActions.ELEVATION_POS:
            self.elevation += 1
        elif viewpoint_action == ViewpointActions.ELEVATION_NEG:
            self.elevation -= 1
        elif viewpoint_action == ViewpointActions.DISTANCE_POS:
            self.distance += 1
        elif viewpoint_action == ViewpointActions.DISTANCE_NEG:
            self.distance -= 1
        elif viewpoint_action == ViewpointActions.X_POS:
            offset_vector = numpy.array([1.,0.,0.,0.])
        elif viewpoint_action == ViewpointActions.X_NEG:
            offset_vector = numpy.array([-1.,0.,0.,0.])
        elif viewpoint_action == ViewpointActions.Y_POS:
            offset_vector = numpy.array([0.,1.,0.,0.])
        elif viewpoint_action == ViewpointActions.Y_NEG:
            offset_vector = numpy.array([0.,-1.,0.,0.])
        
        if offset_vector is not None:
            translate_direction = (
                self.camera_matrix @ offset_vector * self.translate_step_size)
            self.x += translate_direction[0]
            self.y += translate_direction[1]
            self.z += translate_direction[2]
        
        self.azimuth = self.azimuth % self.azimuth_steps
        self.elevation = numpy.clip(
            self.elevation, 0, self.elevation_steps-1)
        self.distance = numpy.clip(self.distance, 0, self.distance_steps-1)
        
        self.set_camera()
        self.compute_observation()
        return self.observation, 0., False, False, {}
    
    def set_camera(self):
        
        # compute the projection matrix
        self.projection = camera.projection_matrix(
            self.field_of_view,
            self.aspect_ratio,
            self.near_clip,
            self.far_clip,
        )
        
        # compute the view matrix
        azimuth = self.azimuth * self.azimuth_step_size
        elevation = (
            self.elevation * self.elevation_step_size +
            self.elevation_range[0]
        )
        distance = (
            self.distance * self.distance_step_size +
            self.distance_range[0]
        )
        self.camera_matrix = camera.azimuthal_parameters_to_matrix(
            azimuth, elevation, 0, distance, 0, 0, self.x, self.y, self.z)
        self.view_matrix = numpy.linalg.inv(self.camera_matrix)
        
        # set the projection and view matrix
        if self.scene_component is not None:
            scene = self.scene_component.brick_scene
            scene.set_projection(self.projection)
            scene.set_view_matrix(self.view_matrix)
    
    # TODO get/set state
    
    def no_op_action(self):
        return 0
