import math
from collections import OrderedDict

import numpy

from gym.spaces import Dict

from ltron.config import Config
from ltron.dataset.info import get_dataset_info
from ltron.gym.envs.ltron_env import LtronEnv
from ltron.gym.components.scene import EmptySceneComponent
from ltron.gym.components.time_step import TimeStepComponent
from ltron.gym.components.loader import DatasetLoaderComponent
from ltron.gym.components.render import (
    ColorRenderComponent, SnapRenderComponent)
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.edit_distance import EditDistance
from ltron.gym.components.assembly import AssemblyComponent
from ltron.gym.components.upright import UprightSceneComponent
from ltron.gym.components.tile import (
    TileMaskComponent, DeduplicateTileMaskComponent)
from ltron.gym.components.cursor import SymbolicCursor, MultiScreenPixelCursor
from ltron.gym.components.brick_inserter import (
    BrickInserter, RandomBrickInserter)
from ltron.gym.components.pick_and_place import PickAndRemove
from ltron.gym.components.rotate import RotateAboutSnap
from ltron.gym.components.phase import PhaseSwitch
from ltron.gym.components.classify_expert import ClassifyExpert
from ltron.gym.components.clear import ClearScene
from ltron.gym.components.viewpoint import ControlledAzimuthalViewpointComponent
from ltron.gym.components.distractor import DistractorComponent

class ClassifyEnvConfig(Config):
    dataset = 'rca'
    
    randomize_viewpoint = True
    
    table_image_height = 256
    table_image_width = 256
    
    tile_color_render = True
    tile_width = 16
    tile_height = 16
    
    num_bricks = 1
    randomize_brick_orientation = True
    num_distractor_bricks = 0
    randomize_distractor_brick_orientation = True
    
    num_distractor_tokens = 0
    distractor_token_classes = 0
    distractor_token_update_frequency = 'step'

class ClassifyEnv(LtronEnv):
    def __init__(
        self,
        config,
        include_expert=False,
        rank=0,
        size=1,
        print_traceback=True,
    ):
        self.include_expert = include_expert
        
        components = OrderedDict()
        
        # get dataset info
        self.dataset_info = get_dataset_info(config.dataset)
        
        self.make_scene_components(config, components)
        self.make_target_components(config, components)
        self.make_action_components(config, components)
        self.make_observation_components(config, components)
        self.make_reward_components(config, components)
        self.make_expert_components(config, components)
        
        super().__init__(
            components,
            combine_action_space='discrete_chain',
            print_traceback=print_traceback,
            early_termination=False,
            expert_component='expert',
        )
    
    def make_scene_components(self, config, components):
        # scenes
        components['table_scene'] = EmptySceneComponent(
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            self.dataset_info['max_instances_per_scene'],
            self.dataset_info['max_edges_per_scene'],
            track_snaps=True,
            collision_checker=False,
        )
        components['estimate_scene'] = EmptySceneComponent(
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            self.dataset_info['max_instances_per_scene'],
            self.dataset_info['max_edges_per_scene'],
            track_snaps=True,
            collision_checker=False,
        )
        
        components['random_insert'] = RandomBrickInserter(
            components['table_scene'],
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            insert_frequency='reset',
            randomize_orientation=config.randomize_brick_orientation,
        )
        
        if config.num_bricks > 1:
            upright = components['table_scene'].brick_scene.upright
            transform = numpy.array([
                [1, 0, 0, 0],
                [0, 1, 0, 100],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]) @ upright
            components['random_insert_2'] = RandomBrickInserter(
                components['table_scene'],
                self.dataset_info['shape_ids'],
                self.dataset_info['color_ids'],
                insert_frequency='reset',
                transform = transform,
                clear_on_insert = False,
                randomize_orientation=config.randomize_brick_orientation,
            )
        
        if config.num_bricks > 2:
            upright = components['table_scene'].brick_scene.upright
            transform = numpy.array([
                [1, 0, 0, 0],
                [0, 1, 0, 100],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]) @ upright
            components['random_insert_2'] = RandomBrickInserter(
                components['table_scene'],
                self.dataset_info['shape_ids'],
                self.dataset_info['color_ids'],
                insert_frequency='reset',
                transform = transform,
                clear_on_insert = False,
                randomize_orientation=config.randomize_brick_orientation,
            )
        
        # time step
        components['step'] = TimeStepComponent(4, observe_step=True)
    
    def make_target_components(self, config, components):
        # target assembly
        components['target_assembly'] = AssemblyComponent(
            components['table_scene'],
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            self.dataset_info['max_instances_per_scene'],
            self.dataset_info['max_edges_per_scene'],
            update_frequency='reset',
            observable=False,
        )
        
        if config.num_distractor_bricks:
            upright = components['table_scene'].brick_scene.upright
            transform = numpy.array([
                [1, 0, 0, 0],
                [0, 1, 0, 100],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]) @ upright
            components['random_insert_2'] = RandomBrickInserter(
                components['table_scene'],
                #self.dataset_info['shape_ids'],
                #self.dataset_info['color_ids'],
                {'2436.dat':1},
                {'1':1},
                insert_frequency='reset',
                transform = transform,
                clear_on_insert = False,
                randomize_orientation=
                    config.randomize_distractor_brick_orientation,
            )
        
        if config.num_distractor_bricks > 1:
            upright = components['table_scene'].brick_scene.upright
            transform = numpy.array([
                [1, 0, 0, 0],
                [0, 1, 0, -100],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]) @ upright
            components['random_insert_3'] = RandomBrickInserter(
                components['table_scene'],
                #self.dataset_info['shape_ids'],
                #self.dataset_info['color_ids'],
                {'2436.dat':1},
                {'1':1},
                insert_frequency='reset',
                transform = transform,
                clear_on_insert = False,
                randomize_orientation=
                    config.randomize_distractor_brick_orientation,
            )
    
    def make_observation_components(self, config, components):
        
        components['table_assembly'] = AssemblyComponent(
            components['table_scene'],
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            self.dataset_info['max_instances_per_scene'],
            self.dataset_info['max_edges_per_scene'],
            update_frequency='step',
            observable=False,
        )
        
        components['estimate_assembly'] = AssemblyComponent(
            components['estimate_scene'],
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            self.dataset_info['max_instances_per_scene'],
            self.dataset_info['max_edges_per_scene'],
            update_frequency = 'step',
            observable=True,
        )
        
        components['table_color_render'] = ColorRenderComponent(
            config.table_image_width,
            config.table_image_height,
            components['table_scene'],
            anti_alias=True,
            observable=False,
        )
        
        if config.tile_color_render:
            components['table_color_tiles'] = DeduplicateTileMaskComponent(
            #components['table_color_tiles'] = TileMaskComponent(
                config.tile_width,
                config.tile_height,
                components['table_color_render'],
                #background=0,
            )
        
        if config.num_distractor_tokens:
            components['distractor'] = DistractorComponent(
                config.num_distractor_tokens,
                config.distractor_token_classes,
                config.distractor_token_update_frequency,
                observable=True,
            )
    
    def make_action_components(self, config, components):
        self.make_visual_action_components(config, components)
        
        components['insert'] = BrickInserter(
            components['estimate_scene'],
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            clear_scene=False,
        )
        
        scene_components = {
            'table':components['table_scene'],
        }
        
        # phase
        components['phase'] = PhaseSwitch(
            scene_components, clear_scenes=True, num_phases=1)
        
    def make_visual_action_components(self, config, components):
        scene_components = {
            'table':components['table_scene'],
        }
        scene_shapes = {
            'table':(config.table_image_height, config.table_image_width),
        }
        viewpoint_distances = {
            'table':320,
        }
        azimuth_steps = 8
        elevation_steps = 2
        for name, scene_component in scene_components.items():
            scene_height, scene_width = scene_shapes[name]
            
            '''
            # Utility Rendering Components -------------------------------------
            components['%s_pos_snap_render'%name] = SnapRenderComponent(
                scene_height,
                scene_width,
                scene_component,
                polarity='+',
                update_frequency='always',#'on_demand',
                observable=False,
            )
            components['%s_neg_snap_render'%name] = SnapRenderComponent(
                scene_height,
                scene_width,
                scene_component,
                polarity='-',
                update_frequency='always',#'on_demand',
                observable=False,
            )
            '''
            
            # Viewpoint --------------------------------------------------------
            elevation_range = [math.radians(-30), math.radians(30)]
            distance_range=[
                viewpoint_distances[name],
                viewpoint_distances[name]
            ]
            if config.randomize_viewpoint:
                start_position = 'uniform'
            else:
                start_position = (0,0,0)
            components['%s_viewpoint'%name] = (
                ControlledAzimuthalViewpointComponent(
                    scene_component,
                    azimuth_steps=azimuth_steps,
                    elevation_range=elevation_range,
                    elevation_steps=elevation_steps,
                    distance_range=distance_range,
                    distance_steps=1,
                    aspect_ratio=scene_width/scene_height, # wuh?
                    start_position=start_position,
                    auto_frame='none', #'reset',
                    frame_button=True,
                )
            )
        
        '''
        components['pick_cursor'] = MultiScreenPixelCursor(
            self.dataset_info['max_instances_per_scene'],
            {'table' : components['table_pos_snap_render']},
            {'table' : components['table_neg_snap_render']},
        )
        '''
    
    def make_reward_components(self, config, components):
        components['reward'] = EditDistance(
            components['target_assembly'],
            components['estimate_assembly'],
            self.dataset_info['shape_ids'],
            pose_penalty=0,
        )
    
    def make_expert_components(self, config, components):
        if self.include_expert:
            #scene_components = {
            #    'table': components['table_scene'],
            #    'estimate': components['estimate_scene'],
            #}
            #assembly_components = {
            #    'table': components['table_assembly'],
            #    'estimate' : components['estimate_assembly'],
            #}
            components['expert'] = ClassifyExpert(
                self,
                #scene_components,
                #components['table_scene'],
                #components['estimate_scene'],
                components['target_assembly'],
                #components['table_assembly'],
                components['estimate_assembly'],
                self.dataset_info['shape_ids'],
            )
    
    def action_to_insert_brick(self, shape, color, pose=None):
        component = self.components['insert']
        s, c = component.actions_to_insert_brick(shape, color)
        return self.action_space.ravel('insert', s, c)
    
    def finish_actions(self):
        current_phase = self.components['phase'].phase
        finish_action = 1
        return [self.action_space.ravel('phase', finish_action)]

