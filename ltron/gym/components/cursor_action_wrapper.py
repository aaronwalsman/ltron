import math

from collections import OrderedDict

from ltron.config import Config
from ltron.gym.envs.ltron_env import LtronEnv
from ltron.gym.components.render import SnapRenderComponent
from ltron.gym.components.cursor import MultiScreenCursor
from ltron.gym.components.pick_and_place import MultiScreenPickAndPlace
from ltron.gym.components.rotation import MultiScreenRotateAboutSnap
from ltron.gym.components.viewpoint import ControlledAzimuthalViewpointComponent
from ltron.gym.components.break_and_make import BreakOnlyPhaseSwitch

class CursorActionWrapperConfig(Config):
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
        table_scene_component,
        hand_scene_component,
        max_instances,
        print_traceback=False,
    ):
        components = OrderedDict()
        
        
        # Utility Rendering Components =========================================
        components['table_pos_snap_render'] = SnapRenderComponent(
            config.table_map_width,
            config.table_map_height,
            table_scene_component,
            polarity='+',
            render_frequency='on_demand',
            observable=False,
        )
        components['table_neg_snap_render'] = SnapRenderComponent(
            config.table_map_width,
            config.table_map_height,
            table_scene_component,
            polarity='-',
            render_frequency='on_demand',
            observable=False,
        )
        components['hand_pos_snap_render'] = SnapRenderComponent(
            config.hand_map_width,
            config.hand_map_height,
            hand_scene_component,
            polarity='+',
            render_frequency='on_demand',
            observable=False,
        )
        components['hand_neg_snap_render'] = SnapRenderComponent(
            config.hand_map_width,
            config.hand_map_height,
            hand_scene_component,
            polarity='-',
            render_frequency='on_demand',
            observable=False,
        )
        
        # Action Spaces ========================================================
        # Viewpoint ------------------------------------------------------------
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
        components['table_viewpoint'] = ControlledAzimuthalViewpointComponent(
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
        components['hand_viewpoint'] = ControlledAzimuthalViewpointComponent(
                hand_scene_component,
                azimuth_steps=config.azimuth_steps,
                elevation_range=elevation_range,
                elevation_steps=config.elevation_steps,
                distance_range=hand_distance_range,
                distance_steps=config.hand_distance_steps,
                aspect_ratio=config.hand_map_width/config.hand_map_height,
                start_position=(0,0,0),
                auto_frame='reset',
                frame_button=True
        )
        
        # Cursors --------------------------------------------------------------
        components['pick_cursor'] = MultiScreenCursor(
            max_instances,
            {'table' : components['table_pos_snap_render'],
             'hand' : components['hand_pos_snap_render']},
            {'table' : components['table_neg_snap_render'],
             'hand' : components['hand_neg_snap_render']},
        )
        components['place_cursor'] = MultiScreenCursor(
            max_instances,
            {'table' : components['table_pos_snap_render'],
             'hand' : components['hand_pos_snap_render']},
            {'table' : components['table_neg_snap_render'],
             'hand' : components['hand_neg_snap_render']},
        )
        
        # Manipulation ---------------------------------------------------------
        components['rotate'] = MultiScreenRotateAboutSnap(
            {'table':table_scene_component, 'hand':hand_scene_component},
            components['pick_cursor'],
            check_collision=config.check_collision,
            allow_snap_flip=True,
        )
        components['pick_and_place'] = MultiScreenPickAndPlace(
            {'table':table_scene_component, 'hand':hand_scene_component},
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
    
    def cursor_where(self, cursor, *args, **kwargs):
        where = self.components[cursor].where(*args, **kwargs)
        return [
            self.action_space.ravel(cursor, w)
            for w in where
        ]
    
    def pick_where(self, *args, **kwargs):
        return self.cursor_where('pick_cursor', *args, **kwargs)
    
    def place_where(self, *args, **kwargs):
        return self.cursor_where('place_cursor', *args, **kwargs)
    
    '''
    def all_viewpoint_actions(self):
        actions = []
        for a, c in (
            self.action_space.name_action_to_chain['table_viewpoint'].items()
        ):
            actions.append(c)
        for a, c in (
            self.action_space.name_action_to_chain['hand_viewpoint'].items()
        ):
            actions.append(c)
        
        return actions
    '''
    
    def all_component_actions(self, component):
        #actions = []
        #for a, c in (
        #    self.action_space.name_action_to_chain[component].items()
        #):
        #    actions.append(c)
        #return actions
        return self.action_space.all_subspace_actions(component)
    
    def pick_and_place_action(self):
        return self.action_space.ravel('pick_and_place',1)
    
    def rotate_action(self, r):
        return self.action_space.ravel('rotate',r)
    
    def finish_action(self):
        phase_action = self.components['phase'].phase + 1
        return self.action_space.ravel('phase', phase_action)
