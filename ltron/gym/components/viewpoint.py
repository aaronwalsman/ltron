from math import radians

import numpy

from gymnasium.spaces import Dict, Discrete, MultiDiscrete

import splendor.camera as camera

from supermecha import SuperMechaComponent

from ltron.constants import DEFAULT_WORLD_BBOX

inf = float('inf')

class ViewpointComponent(SuperMechaComponent):
    def __init__(self,
        scene_component=None,
        azimuth_steps=8,
        azimuth_range=(radians(0.), radians(360.)),
        azimuth_wrap=True,
        elevation_steps=5,
        elevation_range=(radians(-60.), radians(60.)),
        distance_steps=3,
        distance_range=(150.,450.),
        reset_mode='random',
        center_reset_range=((0.,0.,0.),(0.,0.,0.)),
        world_bbox=DEFAULT_WORLD_BBOX,
        translate_step_size=80.,
        field_of_view=radians(60.),
        aspect_ratio=1.,
        near_clip=10.,
        far_clip=50000.,
        frame_action=False,
        observable=True,
        observation_format='coordinates',
    ):
        
        self.scene_component = scene_component
        self.azimuth_steps = azimuth_steps
        self.azimuth_range = azimuth_range
        self.azimuth_wrap = azimuth_wrap
        self.azimuth_step_size = (
            azimuth_range[1] - azimuth_range[0]) / azimuth_steps
        self.elevation_steps = elevation_steps
        self.elevation_range = elevation_range
        self.elevation_step_size = (
            elevation_range[1] - elevation_range[0]) / elevation_steps
        self.distance_steps = distance_steps
        self.distance_range = distance_range
        self.distance_step_size = (
            distance_range[1] - distance_range[0]) / distance_steps
        self.reset_mode = reset_mode
        self.center_reset_range = center_reset_range
        self.world_bbox = world_bbox
        self.translate_step_size = translate_step_size
        self.field_of_view = field_of_view
        self.aspect_ratio = aspect_ratio
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.observable = observable
        self.observation_format = observation_format
        
        # ensure the center starting range is within the center bounds
        assert all(
            [r >= b for r,b in zip(center_reset_range[0], world_bbox[0])])
        assert all(
            [r <= b for r,b in zip(center_reset_range[1], world_bbox[1])])
        
        # make action space
        action_space = {}
        if self.azimuth_steps > 1:
            action_space['azimuth'] = Discrete(3)
        if self.elevation_steps > 1:
            action_space['elevation'] = Discrete(3)
        if self.distance_steps > 1:
            action_space['distance'] = Discrete(3)
        if self.translate_step_size:
            action_space['translate'] = MultiDiscrete((3,3,3))
        if frame_action:
            action_space['frame'] = Discrete(2)
        
        if len(action_space):
            self.action_space = Dict(action_space)
        
        # make observation space
        if self.observable:
            observation_space = {}
            if self.observation_format == 'coord':
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
                        numpy.array(self.world_bbox[0]),
                        numpy.array(self.world_bbox[1]),
                    )
            elif self.observation_format == 'matrix':
                observation_space['camera_matrix'] = SE3Space(self.world_bbox)
            
            if len(observation_space):
                self.observation_space = Dict(observation_space)
    
    def observe(self):
        if self.observable:
            if self.observation_format == 'coordinates':
                self.observation = {}
                self.observation['azimuth'] = self.azimuth
                self.observation['elevation'] = self.elevation
                self.observation['distance'] = self.distance
                self.observation['center'] = numpy.array(
                    [self.x, self.y, self.z])
            
            elif self.observation_format == 'matrix':
                self.observation = {'camera_matrix' : self.camera_matrix}
            
            return self.observation
    
    def reset(self, seed=None, rng=None):
        super().reset(seed=seed, rng=rng)
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
        self.observe()
        return self.observation, None
    
    def step(self, action):
        if 'azimuth' in action:
            self.azimuth += action['azimuth'] - 1
            if self.azimuth_wrap:
                self.azimuth = self.azimuth % self.azimuth_steps
            else:
                self.azimuth = numpy.clip(
                    self.azimuth, 0, self.azimuth_steps-1)
        if 'elevation' in action:
            self.elevation += action['elevation'] - 1
            self.elevation = numpy.clip(
                self.elevation, 0, self.elevation_steps-1)
        if 'distance' in action:
            self.distance += action['distance'] - 1
            self.distance = numpy.clip(self.distance, 0, self.distance_steps-1)
        if 'translate' in action:
            x, y, z = action['translate']
            x = (x-1)*self.translate_step_size
            y = (y-1)*self.translate_step_size
            z = (z-1)*self.translate_step_size
            translate_direction = self.camera_matrix @ [x, y, z, 0]
            self.x += translate_direction[0]
            self.y += translate_direction[1]
            self.z += translate_direction[2]
        
        self.set_camera()
        self.observe()
        return self.observation, 0., False, False, None
    
    def set_camera(self):
        if self.scene_component is not None:
            # get the scene
            scene = self.scene_component.brick_scene
            
            # compute and set the projection matrix
            self.projection = camera.projection_matrix(
                self.field_of_view,
                self.aspect_ratio,
                self.near_clip,
                self.far_clip,
            )
            scene.set_projection(self.projection)
            
            # compute and set the view matrix
            azimuth = (
                self.azimuth * self.azimuth_step_size + self.azimuth_range[0])
            elevation = (
                self.elevation * self.elevation_step_size +
                self.elevation_range[0]
            )
            distance = (
                self.distance * self.distance_step_size +
                self.distance_range[0]
            )
            self.camera_pose = camera.azimuthal_parameters_to_matrix(
                azimuth, elevation, 0, distance, 0, 0, self.x, self.y, self.z)
            self.view_matrix = numpy.linalg.inv(self.camera_pose)
            scene.set_view_matrix(self.view_matrix)
    
    def no_op_action(self):
        action = {}
        if 'azimuth' in self.action_space:
            action['azimuth'] = 1
        if 'elevation' in self.action_space:
            action['elevation'] = 1
        if 'distance' in self.action_space:
            action['distance'] = 1
        if 'translate' in self.action_space:
            action['translate'] = (1,1,1)
        if 'frame' in self.action_space:
            action['frame'] = 0
        
        return action
