from math import radians
from collections import OrderedDict

from gymnasium.spaces import Discrete

from steadfast import Config

from supermecha import SuperMechaContainer, SuperMechaComponentSwitch

from ltron.constants import DEFAULT_WORLD_BBOX
from ltron.gym.components import (
    ViewpointComponent,
    SnapCursorComponent,
    CursorRemoveBrickComponent,
    CursorPickAndPlaceComponent,
    DoneComponent,
    SnapMaskRenderComponent,
    SnapIslandRenderComponent,
    OverlayBrickComponent,
)

class VisualInterfaceConfig(Config):
    # image geometry
    image_height = 256
    image_width = 256
    
    # collisions
    check_collision = True
    
    # world size
    world_bbox = DEFAULT_WORLD_BBOX
    
    # viewpoint
    viewpoint_azimuth_steps = 16
    viewpoint_elevation_steps = 5
    viewpoint_elevation_range = (radians(-60.), radians(60.))
    viewpoint_distance_steps = 3
    viewpoint_distance_range = (150.,450.)
    viewpoint_reset_mode = 'random'
    viewpoint_center_reset = ((0.,0.,0.),(0.,0.,0.))
    viewpoint_translate_step_size = 40.
    viewpoint_field_of_view = radians(60.)
    viewpoint_near_clip = 10.
    viewpoint_far_clip = 50000.
    viewpoint_observable = True
    
    # cursor
    tile_cursor = True
    cursor_tile_height = 16
    cursor_tile_width = 16
    cursor_observe_selected = False

class VisualInterface(SuperMechaContainer):
    def __init__(self,
        config,
        scene_component,
        train=True,
    ):
        components = OrderedDict()
        mode_components = OrderedDict()
        
        # cursor
        pos_snap_render_component = SnapMaskRenderComponent(
            scene_component,
            config.image_height,
            config.image_width,
            polarity='+',
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            observable=False,
        )
        neg_snap_render_component = SnapMaskRenderComponent(
            scene_component,
            config.image_height,
            config.image_width,
            polarity='-',
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            observable=False,
        )
        
        components['cursor'] = SnapCursorComponent(
            scene_component,
            pos_snap_render_component,
            neg_snap_render_component,
            config.image_height,
            config.image_width,
            train=train,
        )
        
        # table viewpoint
        aspect_ratio = config.image_width / config.image_height
        mode_components['table_viewpoint'] = ViewpointComponent(
            scene_component=scene_component,
            azimuth_steps=config.viewpoint_azimuth_steps,
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
            observable=config.viewpoint_observable,
        )
        
        # hand viewpoint
        mode_components['hand_viewpoint'] = ViewpointComponent(
            scene_component=None,
            azimuth_steps=config.viewpoint_azimuth_steps,
            elevation_steps=config.viewpoint_elevation_steps,
            elevation_range=config.viewpoint_elevation_range,
            distance_steps=config.viewpoint_distance_steps,
            distance_range=config.viewpoint_distance_range,
            reset_mode=config.viewpoint_reset_mode,
            world_bbox=config.world_bbox, #TODO: should be a separate bbox
            allow_translate=False,
            #translate_step_size=config.viewpoint_translate_step_size,
            field_of_view=config.viewpoint_field_of_view,
            observable=config.viewpoint_observable,
        )
        
        # overlay brick
        mode_components['overlay_brick'] = OverlayBrickComponent(
            scene_component,
            mode_components['table_viewpoint'],
            mode_components['hand_viewpoint'],
        )
        
        # pick and place
        mode_components['pick_and_place'] = CursorPickAndPlaceComponent(
            scene_component,
            components['cursor'],
            overlay_brick_component = mode_components['overlay_brick'],
            check_collision=config.check_collision,
        )
        
        # removal
        mode_components['remove'] = CursorRemoveBrickComponent(
            scene_component,
            components['cursor'],
            check_collision=config.check_collision,
        )
        
        # done
        #mode_components['done'] = DoneComponent()
        
        # make the mode switch
        components['primitives'] = SuperMechaComponentSwitch(
            mode_components, switch_name='mode')
        
        components['pos_snap_render'] = pos_snap_render_component
        components['neg_snap_render'] = neg_snap_render_component
        components['pos_equivalence'] = SnapIslandRenderComponent(
            scene_component,
            pos_snap_render_component,
            config.image_height,
            config.image_width,
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            observable=True,
        )
        components['neg_equivalence'] = SnapIslandRenderComponent(
            scene_component,
            neg_snap_render_component,
            config.image_height,
            config.image_width,
            update_on_init=False,
            update_on_reset=True,
            update_on_step=True,
            observable=True,
        )
        
        super().__init__(components)
