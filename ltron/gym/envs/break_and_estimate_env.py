import math
from collections import OrderedDict

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
from ltron.gym.components.tile import DeduplicateTileMaskComponent
from ltron.gym.components.cursor import SymbolicCursor, MultiScreenPixelCursor
from ltron.gym.components.brick_inserter import BrickInserter
from ltron.gym.components.pick_and_place import PickAndRemove
from ltron.gym.components.rotate import RotateAboutSnap
from ltron.gym.components.phase import PhaseSwitch
from ltron.gym.components.estimate_expert import EstimateExpert
from ltron.gym.components.clear import ClearScene
from ltron.gym.components.viewpoint import ControlledAzimuthalViewpointComponent

class BreakAndEstimateEnvConfig(Config):
    dataset = 'rca'
    split = '2_2_train'
    subset = None
    
    max_episode_length = 32
    
    shuffle = True
    shuffle_buffer = 100
    repeat = True
    
    randomize_colors = False
    randomize_viewpoint = True
    randomize_starting_cursor = True
    
    table_image_height = 256
    table_image_width = 256
    
    tile_color_render = True
    tile_width = 16
    tile_height = 16
    
    check_collision = True
    
    # expert
    max_instructions = 2048
    shuffle_instructions = True
    expert_always_add_viewpoint_actions = False
    early_termination = False

class BreakAndEstimateEnv(LtronEnv):
    def __init__(
        self,
        config,
        include_expert=False,
        rank=0,
        size=1,
        print_traceback=True,
    ):
        self.include_expert = include_expert
        self.rank = rank
        self.size = size
        
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
            early_termination=config.early_termination * include_expert,
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
            collision_checker=config.check_collision,
        )
        components['estimate_scene'] = EmptySceneComponent(
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            self.dataset_info['max_instances_per_scene'],
            self.dataset_info['max_edges_per_scene'],
            track_snaps=True,
            collision_checker=config.check_collision,
        )
        
        # loader
        components['dataset'] = DatasetLoaderComponent(
            components['table_scene'],
            config.dataset,
            config.split,
            subset=config.subset,
            rank=self.rank,
            size=self.size,
            shuffle=config.shuffle,
            shuffle_buffer=config.shuffle_buffer,
            repeat=config.repeat,
        )
        
        # uprightify
        components['upright'] = UprightSceneComponent(
            scene_component = components['table_scene'])
        
        # time step
        components['step'] = TimeStepComponent(
            config.max_episode_length, observe_step=True)
        
        # color randomization
        if config.randomize_colors:
            components['color_randomization'] = RandomizeColorsComponent(
                self.dataset_info['color_ids'],
                components['table_scene'],
                randomize_frequency='reset',
            )
    
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
                config.tile_width,
                config.tile_height,
                components['table_color_render'],
            )
    
    def make_action_components(self, config, components):
        self.make_visual_action_components(config, components)
        
        components['insert'] = BrickInserter(
            components['estimate_scene'],
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            include_pose=False, # DO THIS NEXT
            clear_scene=False,
        )
        
        scene_components = {
            'table':components['table_scene'],
        }
        #components['pick_and_remove'] = PickAndRemove(
        #    scene_components,
        #    components['pick_cursor'],
        #    check_collision=config.check_collision,
        #)
        
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
                    auto_frame='reset',
                    frame_button=True,
                )
            )
        
        components['pick_cursor'] = MultiScreenPixelCursor(
            self.dataset_info['max_instances_per_scene'],
            {'table' : components['table_pos_snap_render']},
            {'table' : components['table_neg_snap_render']},
        )
    
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
            components['expert'] = EstimateExpert(
                self,
                #scene_components,
                components['table_scene'],
                components['estimate_scene'],
                components['target_assembly'],
                components['table_assembly'],
                components['estimate_assembly'],
                self.dataset_info['shape_ids'],
                max_instructions=config.max_instructions,
                shuffle_instructions=config.shuffle_instructions,
                always_add_viewpoint_actions=
                    config.expert_always_add_viewpoint_actions,
                terminate_on_empty=True,
            )
    
    def get_pick_snap(self):
        return self.components['pick_cursor'].get_selected_snap()
    
    def finish_actions(self):
        current_phase = self.components['phase'].phase
        finish_action = 1
        return [self.action_space.ravel('phase', finish_action)]
    
    def actions_to_select_snap(self, cursor, *args, **kwargs):
        actions = self.components[cursor].actions_to_select_snap(
            *args, **kwargs)
        result = [
            self.action_space.ravel(cursor, *a)
            for a in actions
        ]
        return result
    
    def actions_to_pick_snap(self, *args, **kwargs):
        return self.actions_to_select_snap('pick_cursor', *args, **kwargs)
    
    def actions_to_deselect(self, cursor, *args, **kwargs):
        actions = self.components[cursor].actions_to_deselect(*args, **kwargs)
        return [self.action_space.ravel(cursor, *a) for a in actions]
    
    def actions_to_deselect_pick(self, *args, **kwargs):
        return self.actions_to_deselect('pick_cursor', *args, **kwargs)
    
    def all_component_actions(self, component, include_no_op=True):
        start, stop = self.action_space.name_range(component)
        if include_no_op:
            return list(range(start, stop))
        else:
            return [r for r in range(start, stop)
                if r != start + self.components[component].no_op_action()]
    
    def pick_and_remove_action(self, p):
        return self.action_space.ravel('pick_and_remove', p)
    
    def actions_to_insert_brick(self, shape, color):
        s, c = self.components['insert'].actions_to_insert_brick(shape, color)
        return self.action_space.ravel('insert', s, c)
