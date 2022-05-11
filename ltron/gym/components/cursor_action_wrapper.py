import math

from collections import OrderedDict

from ltron.config import Config
from ltron.gym.envs.ltron_env import LtronEnv
from ltron.gym.components.render import SnapRenderComponent
from ltron.gym.components.cursor import SymbolicCursor, MultiViewCursor
from ltron.gym.components.pick_and_place import MultiScenePickAndPlace
from ltron.gym.components.rotation import MultiSceneRotateAboutSnap
from ltron.gym.components.viewpoint import ControlledAzimuthalViewpointComponent
from ltron.gym.components.break_and_make import BreakOnlyPhaseSwitch

class CursorActionWrapperConfig(Config):
    cursor_mode = 'spatial'
    
    table_map_height = 64
    table_map_width = 64
    hand_map_height = 24
    hand_map_width = 24
    
    check_collision = True
    
    randomize_viewpoint = True
    table_min_distance = 320
    table_max_distance = 320
    table_distance_steps = 1
    hand_min_distance = 180
    hand_max_distance = 180
    hand_distance_steps = 1
    azimuth_steps = 8
    elevation_steps = 2

class CursorActionWrapper(LtronEnv):
    def __init__(
        self,
        config,
        #table_scene_component,
        #hand_scene_component,
        scene_components,
        max_instances,
        assembly_components=None,
        print_traceback=False,
    ):
        components = OrderedDict()
        
        # spatial-specific components ==========================================
        if config.cursor_mode == 'spatial':
            
            for name, scene_component in scene_components:
                
                # Utility Rendering Components ---------------------------------
                components['%s_pos_snap_render'%name] = SnapRenderComponent(
                    scene_height, # where does this come from?
                    scene_width, # where does this come from?
                    scene_component,
                    polarity='+',
                    render_frequency='on_demand',
                    observable=False,
                )
                components['%s_neg_snap_render'%name] = SnapRenderComponent(
                    scene_height, # where does this come from?
                    scene_width, # where does this come from?
                    scene_component,
                    polarity='-',
                    render_frequency='on_demand',
                    observable=False,
                )
        
                # Viewpoint ----------------------------------------------------
                elevation_range = [math.radians(-30), math.radians(30)]
                distance_range=[
                    config.min_viewpoint_distance,
                    config.max_viewpoint_distance
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
                        aspect_ratio=scene_width/scene_height, # wuh?
                        start_position=start_position,
                        auto_frame='reset',
                        frame_button=True,
                    )
                )
                
            '''
            elevation_range = [math.radians(-30), math.radians(30)]
            # TODO: make this correct
            table_distance_range=[
                config.table_min_distance, config.table_max_distance]
            hand_distance_range=[
                config.hand_min_distance, config.hand_max_distance]
            if config.randomize_viewpoint:
                start_position='uniform'
            else:
                start_position=(0,0,0)
            components['table_viewpoint'] = (
                ControlledAzimuthalViewpointComponent(
                    table_scene_component,
                    azimuth_steps=config.azimuth_steps,
                    elevation_range=elevation_range,
                    elevation_steps=config.elevation_steps,
                    distance_range=table_distance_range,
                    distance_steps=config.table_distance_steps,
                    aspect_ratio=config.table_map_width/config.table_map_height,
                    start_position=start_position,
                    auto_frame='reset',
                    frame_button=True,
                )
            )
            components['hand_viewpoint'] = (
                ControlledAzimuthalViewpointComponent(
                    hand_scene_component,
                    azimuth_steps=config.azimuth_steps,
                    elevation_range=elevation_range,
                    elevation_steps=config.elevation_steps,
                    distance_range=hand_distance_range,
                    distance_steps=config.hand_distance_steps,
                    aspect_ratio=config.hand_map_width/config.hand_map_height,
                    start_position=(0,0,0),
                    auto_frame='reset',
                    frame_button=True,
                )
            )
            '''
            
            # Cursors ----------------------------------------------------------
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
        
        # symbolic-specific components =========================================
        elif config.cursor_mode == 'symbolic':
            
            # Cursors ----------------------------------------------------------
            components['pick_cursor'] = SymbolicCursor(
                assembly_components,
            )
            components['place_cursor'] = SymbolicCursor(
                assembly_components,
            )
        
        else:
            raise ValueError(
                'cursor_mode must be either "visual" or "symbolic"'
            )
        
        # Manipulation ---------------------------------------------------------
        components['rotate'] = MultiSceneRotateAboutSnap(
            scene_components,
            components['pick_cursor'],
            check_collision=config.check_collision,
            allow_snap_flip=True,
        )
        components['pick_and_place'] = MultiScenePickAndPlace(
            scene_components,
            components['pick_cursor'],
            components['place_cursor'],
            check_collision=config.check_collision,
        )
        
        # Finished -------------------------------------------------------------
        components['phase'] = BreakOnlyPhaseSwitch()
        
        super().__init__(
            components,
            combine_action_space='discrete_chain',
            print_traceback=print_traceback,
        )
    
    def get_selected_pick_snap(self):
        return self.components['pick_cursor'].get_selected_snap()
    
    def get_selected_place_snap(self):
        return self.components['place_cursor'].get_selected_snap()
    
    def actions_to_select_snap(self, cursor, *args, **kwargs):
        actions = self.components[cursor].actions_to_select_snap(
            *args, **kwargs)
        return [
            self.action_space.ravel(cursor, a)
            for a in actions
        ]
    
    def actions_to_pick_snap(self, *args, **kwargs):
        return self.actions_to_select_snap('pick_cursor', *args, **kwargs)
    
    def actions_to_place_snap(self, *args, **kwargs):
        return self.actions_to_select_snap('place_cursor', *args, **kwargs)
    
    def all_component_actions(self, component):
        return self.action_space.all_subspace_actions(component)
    
    def pick_and_place_action(self):
        return self.action_space.ravel('pick_and_place',1)
    
    def rotate_action(self, r):
        return self.action_space.ravel('rotate',r)
    
    def finish_action(self):
        phase_action = self.components['phase'].phase + 1
        return self.action_space.ravel('phase', phase_action)
