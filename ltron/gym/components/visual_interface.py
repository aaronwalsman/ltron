from math import radians
from collections import OrderedDict

from steadfast import Config

from supermecha import SuperMechaContainer

from ltron.constants import DEFAULT_WORLD_BBOX
from ltron.gym.components import (
    ViewpointComponent,
    ColorRenderComponent,
    SnapRenderComponent,
    PickAndPlaceCursor,
    TiledScreenCursor,
    CursorRemoveBrickComponent,
)

class VisualInterfaceConfig(Config):
    # screen geometry
    screen_height = 256
    screen_width = 256
    
    tile_image = True
    tile_height = 16
    tile_width = 16
    
    # collisions
    check_collision = True
    
    # world size
    world_bbox = DEFAULT_WORLD_BBOX
    
    # viewpoint
    viewpoint_azimuth_steps = 8
    viewpoint_azimuth_range = (radians(0.), radians(360.))
    viewpoint_azimuth_wrap = True
    viewpoint_elevation_steps = 5
    viewpoint_elevation_range = (radians(-60.), radians(60.))
    viewpoint_distance_steps = 3
    viewpoint_distance_range = (150.,450.)
    viewpoint_reset_mode = 'random'
    viewpoint_center_reset = ((0.,0.,0.),(0.,0.,0.))
    viewpoint_translate_step_size = 80.
    viewpoint_field_of_view = radians(60.)
    viewpoint_near_clip = 10.
    viewpoint_far_clip = 50000.
    viewpoint_frame_action = False
    viewpoint_observable = True
    viewpoint_observation_format = 'coordinates'
    
    # cursor
    tile_cursor = True
    cursor_tile_height = 16
    cursor_tile_width = 16
    cursor_observe_selected = False
    

class VisualInterface(SuperMechaContainer):
    def __init__(self,
        config,
        scene_component,
        max_instances_per_scene,
        include_manipulation=True,
        include_floating_pane=True,
        include_brick_removal=True,
        include_viewpoint=True,
    ):
        components = OrderedDict()
        
        if include_viewpoint:
            
            # viewpoint
            aspect_ratio = config.screen_width / config.screen_height
            components['viewpoint'] = ViewpointComponent(
                scene_component=scene_component,
                azimuth_steps=config.viewpoint_azimuth_steps,
                azimuth_range=config.viewpoint_azimuth_range,
                azimuth_wrap=config.viewpoint_azimuth_wrap,
                elevation_steps=config.viewpoint_elevation_steps,
                elevation_range=config.viewpoint_elevation_range,
                distance_steps=config.viewpoint_distance_steps,
                distance_range=config.viewpoint_distance_range,
                reset_mode=config.viewpoint_reset_mode,
                center_reset_range=config.viewpoint_center_reset,
                world_bbox=config.world_bbox,
                translate_step_size=config.viewpoint_translate_step_size,
                field_of_view=config.viewpoint_field_of_view,
                aspect_ratio=aspect_ratio,
                near_clip=config.viewpoint_near_clip,
                far_clip=config.viewpoint_far_clip,
                frame_action=config.viewpoint_frame_action,
                observable=config.viewpoint_observable,
                observation_format=config.viewpoint_observation_format
            )
        
        if include_manipulation:
            if include_viewpoint and include_floating_pane:
                
                # floating viewpoint
                components['floating_viewpoint'] = ViewpointComponent(
                    scene_component=None,
                    azimuth_steps=config.floating_azimuth_steps,
                    azimuth_range=(radians(0), radians(360)),
                    azimuth_wrap=True,
                    elevation_steps=config.floating_elevation_steps,
                    elevation_range=config.floating_elevation_range,
                    distance_steps=config.floating_distance_steps,
                    distance_range=config.floating_distance_range,
                    reset_mode='random',
                    world_bbox=config.world_bbox,
                    translate_step_size=config.viewpoint_translate_step_size,
                    field_of_view=config.viewpoint_field_of_view,
                    aspect_ratio=aspect_ratio,
                    near_clip=config.viewpoint_near_clip,
                    far_clip=config.viewpoint_far_clip,
                    frame_action=config.viewpoint_frame_action,
                    observable=config.viewpoint_observable,
                    observation_format=config.viewpoint_observation_format
                )
            
            # snap render
            components['snap_render'] = SnapRenderComponent(
                scene_component,
                config.screen_height,
                config.screen_width,
                update_on_init=False,
                update_on_reset=False,
                update_on_step=False,
                cache_observation=False,
                observable=False,
            )
            
            # cursor
            if config.tile_cursor:
                CursorClass = TiledScreenCursor
                cursor_args = {
                    'tile_height' : config.cursor_tile_height,
                    'tile_width' : config.cursor_tile_width,
                }
            else:
                CursorClass = ScreenCursor
                cursor_args = {}
            
            pick_cursor, place_cursor = [CursorClass(
                scene_component,
                components['snap_render'],
                max_instances_per_scene,
                observe_selected=config.cursor_observe_selected,
                **cursor_args,
            ) for _ in range(2)]
            components['cursor'] = PickAndPlaceCursor(pick_cursor, place_cursor)
            
            # removal
            if include_brick_removal:
                components['remove'] = CursorRemoveBrickComponent(
                    scene_component,
                    pick_cursor,
                    check_collision=config.check_collision,
                )
        
        # color render
        components['color_render'] = ColorRenderComponent(
            scene_component,
            config.screen_height,
            config.screen_width,
            anti_alias=True,
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            cache_observation=True,
            observable=True,
        )
        
        super().__init__(components)
