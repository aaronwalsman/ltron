from collections import OrderedDict

from ltron.config import Config
from ltron.dataset.paths import get_dataset_info
from ltron.gym.envs.ltron_env import LtronEnv
from ltron.gym.components.scene import EmptySceneComponent
from ltron.gym.components.episode import EpisodeLengthComponent
from ltron.gym.components.loader import DatasetLoaderComponent
from ltron.gym.components.render import ColorRenderComponent
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.edit_distance import EditDistance
from ltron.gym.components.assembly import AssemblyComponent
from ltron.gym.components.upright import UprightSceneComponent
from ltron.gym.components.tile import DeduplicateTileMaskComponent
from ltron.gym.components.cursor import SymbolicCursor
from ltron.gym.components.brick_inserter import BrickInserter
from ltron.gym.components.pick_and_place import PickAndPlace
from ltron.gym.components.rotate import RotateAboutSnap
from ltron.gym.components.phase import PhaseSwitch
from ltron.gym.components.build_expert import BuildExpert
from ltron.gym.components.clear import ClearScene

class BreakAndMakeEnvConfig(Config):
    dataset = 'rca'
    split = '2_2_train'
    subset = None
    
    target_mode = 'interactive'
    observation_mode = 'visual'
    action_mode = 'visual'
    
    max_episode_length = 32
    
    dataset_sample_mode = 'uniform'
    
    randomize_colors = False
    
    table_image_height = 256
    table_image_width = 256
    hand_image_height = 96
    hand_image_width = 96
    
    tile_color_render = True
    tile_width = 16
    tile_height = 16
    
    check_collision = True
    
    # expert
    max_instructions = 128
    expert_always_add_viewpoint_actions = False

class BreakAndMakeEnv(LtronEnv):
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
        components['hand_scene'] = EmptySceneComponent(
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
            sample_mode=config.dataset_sample_mode,
        )

        # uprightify
        components['upright'] = UprightSceneComponent(
            scene_component = components['table_scene'])

        # time step
        components['step'] = EpisodeLengthComponent(
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
            observable=(config.observation_mode == 'symbolic'),
        )
    
    def make_observation_components(self, config, components):
        if config.target_mode != 'interactive':
            scene_components = {
                'table':components['table_scene'],
                'hand':components['hand_scene'],
            }
            components['clear'] = ClearScene(scene_components)
        
        components['table_assembly'] = AssemblyComponent(
            components['table_scene'],
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            self.dataset_info['max_instances_per_scene'],
            self.dataset_info['max_edges_per_scene'],
            update_frequency='step',
            observable=(config.observation_mode == 'symbolic'),
        )
        
        components['hand_assembly'] = AssemblyComponent(
            components['hand_scene'],
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
            self.dataset_info['max_instances_per_scene'],
            self.dataset_info['max_edges_per_scene'],
            update_frequency = 'step',
            observable = (config.observation_mode == 'symbolic'),
        )
        
        if config.observation_mode == 'symbolic':
            self.make_symbolic_observation_components(config, components)
        
        elif config.observation_mode == 'visual':
            self.make_visual_observation_components(config, components)
            
        elif config.observation_mode == 'none':
            pass
        
        else:
            raise ValueError(
                'observation_mode must be in (symbolic, visual, none) '
                'got: "%s"'%config.observation_mode)
    
    def make_symbolic_observation_components(self, config, components):
        pass
    
    def make_visual_observation_components(self, config, components):
        table_color_render = ColorRenderComponent(
            config.table_image_width,
            config.table_image_height,
            components['table_scene'],
            anti_alias=True,
            observable=False,
        )
        hand_color_render = ColorRenderComponent(
            config.hand_image_width,
            config.hand_image_height,
            components['hand_scene'],
            anti_alias=True,
            observable=False,
        )
        
        if config.tile_color_render:
            components['table_color_tiles'] = DeduplicateTileMaskComponent(
                config.tile_width,
                config.tile_height,
                table_color_render,
            )
            components['hand_color_tiles'] = DeduplicateTileMaskComponent(
                config.tile_width,
                config.tile_height,
                hand_color_render,
            )
        
        else:
            components['table_color_render'] = table_color_render
            components['hand_color_render'] = hand_color_render
    
    def make_action_components(self, config, components):
        if config.action_mode == 'symbolic':
            self.make_symbolic_action_components(config, components)
        
        elif config.action_mode == 'visual':
            self.make_visual_action_components(config, components)
        
        elif config.action_mode == 'se3':
            pass
        
        else:
            raise ValueError(
                'action_mode must be in (symbolic, visual, se3) '
                'got: "%s"'%config.action_mode)
        
        components['insert'] = BrickInserter(
            components['hand_scene'],
            self.dataset_info['shape_ids'],
            self.dataset_info['color_ids'],
        )
        scene_components = {
            'table':components['table_scene'],
            'hand':components['hand_scene']
        }
        components['rotate'] = RotateAboutSnap(
            scene_components,
            components['pick_cursor'],
            check_collision=config.check_collision,
            allow_snap_flip=True,
        )
        components['pick_and_place'] = PickAndPlace(
            scene_components,
            components['pick_cursor'],
            components['place_cursor'],
            max_instances_per_scene=
                self.dataset_info['max_instances_per_scene'],
            check_collision=config.check_collision,
        )
        
        # phase
        if config.target_mode == 'interactive':
            num_phases = 2
        else:
            num_phases = 1
        
        components['phase'] = PhaseSwitch(
            scene_components, clear_scenes=True, num_phases=num_phases)
        
    
    def make_symbolic_action_components(self, config, components):
        scene_components = {
            'table':components['table_scene'],
            'hand':components['hand_scene'],
        }
        
        components['pick_cursor'] = SymbolicCursor(
            scene_components, self.dataset_info['max_instances_per_scene'])
        components['place_cursor'] = SymbolicCursor(
            scene_components, self.dataset_info['max_instances_per_scene'])
    
    def make_visual_action_components(self, config, components):
        for name, scene_component in scene_components.items():
            scene_height, scene_width = scene_shapes[name]
            
            # Utility Rendering Components ---------------------------------
            components['%s_pos_snap_render'%name] = SnapRenderComponent(
                scene_height, # where does this come from?
                scene_width, # where does this come from?
                scene_component,
                polarity='+',
                update_frequency='on_demand',
                observable=False,
            )
            components['%s_neg_snap_render'%name] = SnapRenderComponent(
                scene_height, # where does this come from?
                scene_width, # where does this come from?
                scene_component,
                polarity='-',
                update_frequency='on_demand',
                observable=False,
            )

            # Viewpoint ----------------------------------------------------
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
                    azimuth_steps=config.azimuth_steps,
                    elevation_range=elevation_range,
                    elevation_steps=config.elevation_steps,
                    distance_range=distance_range,
                    distance_steps=1,
                    aspect_ratio=scene_width/scene_height, # wuh?
                    start_position=start_position,
                    auto_frame='reset',
                    frame_button=True,
                )
            )
        
        components['pick_cursor'] = MultiViewCursor(
            max_instances,
            {'table' : components['table_pos_snap_render'],
             'hand' : components['hand_pos_snap_render']},
            {'table' : components['table_neg_snap_render'],
             'hand' : components['hand_neg_snap_render']},
        )
        components['place_cursor'] = MultiViewCursor(
            max_instances,
            {'table' : components['table_pos_snap_render'],
             'hand' : components['hand_pos_snap_render']},
            {'table' : components['table_neg_snap_render'],
             'hand' : components['hand_neg_snap_render']},
        )
    
    def make_reward_components(self, config, components):
        components['reward'] = EditDistance(
            components['target_assembly'],
            components['table_assembly'],
            self.dataset_info['shape_ids'],
        )
    
    def make_expert_components(self, config, components):
        if self.include_expert:
            scene_components = {
                'table': components['table_scene'],
                'hand': components['hand_scene'],
            }
            assembly_components = {
                'table': components['table_assembly'],
                'hand' : components['hand_assembly'],
            }
            components['expert'] = BuildExpert(
                self,
                scene_components,
                components['target_assembly'],
                assembly_components,
                'table',
                self.dataset_info['shape_ids'],
                config.max_instructions,
                config.expert_always_add_viewpoint_actions,
            )
    
    def get_pick_snap(self):
        return self.components['pick_cursor'].get_selected_snap()
    
    def get_place_snap(self):
        return self.components['place_cursor'].get_selected_snap()
    
    def finish_actions(self):
        current_phase = self.components['phase'].phase
        return [self.action_space.ravel('phase', current_phase+1)]
    
    def actions_to_select_snap(self, cursor, *args, **kwargs):
        actions = self.components[cursor].actions_to_select_snap(
            *args, **kwargs)
        return [
            self.action_space.ravel(cursor, *a)
            for a in actions
        ]
    
    def actions_to_pick_snap(self, *args, **kwargs):
        return self.actions_to_select_snap('pick_cursor', *args, **kwargs)
    
    def actions_to_place_snap(self, *args, **kwargs):
        return self.actions_to_select_snap('place_cursor', *args, **kwargs)
    
    def actions_to_deselect(self, cursor):
        actions = self.components[cursor].deselect_actions(*args, **kwargs)
        return [self.action_space.ravel(cursor, *a) for a in actions]
    
    def actions_to_deselect_pick(self):
        return self.actions_to_deselect('pick_cursor')
    
    def actions_to_deselect_place(self):
        return self.actions_to_deselect('place_cursor')
    
    def all_component_actions(self, component):
        return list(range(*self.action_space.name_range(component)))
    
    def rotate_actions(self, r):
        return self.action_space.ravel('rotate', r)
    
    def pick_and_place_action(self):
        return self.action_space.ravel('pick_and_place')
    
    def actions_to_insert_brick(self, shape, color):
        s, c = self.components['insert'].actions_to_insert_brick(shape, color)
        return self.action_space.ravel('insert', s, c)
