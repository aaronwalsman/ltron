from math import radians
from collections import OrderedDict

from gymnasium.spaces import Discrete

from steadfast import Config

from supermecha import SuperMechaContainer, SuperMechaComponentSwitch

from ltron.constants import DEFAULT_WORLD_BBOX
from ltron.gym.components import (
    FixedViewpointComponent,
    ViewpointComponent,
    SnapCursorComponent,
    CursorRemoveBrickComponent,
    CursorPickAndPlaceComponent,
    CursorRotateSnapAboutAxisComponent,
    CursorOrthogonalCameraSpaceRotationComponent,
    CursorOrthogonalCameraSpaceTranslateComponent,
    DoneComponent,
    SnapMaskRenderComponent,
    InsertBrickComponent,
    BreakAndMakePhaseSwitchComponent,
    AssembleStepComponent,
)

class VisualInterfaceConfig(Config):
    # image geometry
    image_height = 256
    image_width = 256
    
    # collisions
    check_collision = True
    
    # world size
    world_bbox = DEFAULT_WORLD_BBOX
    
    # components
    include_viewpoint = True
    include_pick_and_place = True
    include_rotate = True
    include_translate = True
    include_remove = True
    include_insert = True
    include_done = True
    include_phase = False
    include_assemble_step = False
    
    # viewpoint
    viewpoint_azimuth_steps = 16
    viewpoint_elevation_steps = 5
    viewpoint_elevation_range = (radians(-60.), radians(60.))
    viewpoint_distance_steps = 3
    viewpoint_distance_range = (300.,600.)
    viewpoint_reset_mode = 'random'
    viewpoint_center_reset = ((0.,0.,0.),(0.,0.,0.))
    viewpoint_translate_step_size = 40.
    viewpoint_field_of_view = radians(60.)
    viewpoint_near_clip = 10.
    viewpoint_far_clip = 50000.
    viewpoint_observable = True
    viewpoint_orientation_jiggle = False
    viewpoint_translation_jiggle = False
    viewpoint_center_on_reset = False
    viewpoint_fixed_elevation = 1
    
    truncate_on_failure = False

def make_visual_interface(
    config,
    scene_component,
    target_assembly_component=None,
    train=True,
):
    components = OrderedDict()
    action_primitives = OrderedDict()
    
    # snap render
    pos_snap_render_component = SnapMaskRenderComponent(
        scene_component,
        config.image_height,
        config.image_width,
        polarity='+',
        update_on_init=False,
        update_on_reset=True,
        update_on_step=True,
        observable=True,
    )
    neg_snap_render_component = SnapMaskRenderComponent(
        scene_component,
        config.image_height,
        config.image_width,
        polarity='-',
        update_on_init=False,
        update_on_reset=True,
        update_on_step=True,
        observable=True,
    )
    
    # cursor
    components['cursor'] = SnapCursorComponent(
        scene_component,
        pos_snap_render_component,
        neg_snap_render_component,
        config.image_height,
        config.image_width,
    )
    
    # viewpoint
    if config.include_viewpoint:
        aspect_ratio = config.image_width / config.image_height
        action_primitives['viewpoint'] = ViewpointComponent(
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
            orientation_jiggle=config.viewpoint_orientation_jiggle,
            translation_jiggle=config.viewpoint_translation_jiggle,
            center_on_reset=config.viewpoint_center_on_reset
        )
    else:
        aspect_ratio = config.image_width / config.image_height
        components['viewpoint'] = FixedViewpointComponent(
            scene_component=scene_component,
            azimuth=5,
            azimuth_steps=config.viewpoint_azimuth_steps,
            elevation=config.viewpoint_fixed_elevation,
            elevation_steps=config.viewpoint_elevation_steps,
            elevation_range=config.viewpoint_elevation_range,
            distance=1,
            distance_steps=config.viewpoint_distance_steps,
            distance_range=config.viewpoint_distance_range,
            world_bbox=config.world_bbox,
            field_of_view=config.viewpoint_field_of_view,
            aspect_ratio=aspect_ratio,
            near_clip=config.viewpoint_near_clip,
            far_clip=config.viewpoint_far_clip,
            orientation_jiggle=config.viewpoint_orientation_jiggle,
            translation_jiggle=config.viewpoint_translation_jiggle,
            center_on_reset=config.viewpoint_center_on_reset,
        )
    
    # pick and place
    if config.include_pick_and_place:
        action_primitives['pick_and_place'] = CursorPickAndPlaceComponent(
            scene_component,
            components['cursor'],
            check_collision=config.check_collision,
            truncate_on_failure=config.truncate_on_failure,
        )
    
    # rotate
    if config.include_rotate:
        #action_primitives['rotate'] = CursorRotateSnapComponent(
        #    scene_component,
        #    components['cursor'],
        #    check_collision=config.check_collision,
        #)
        action_primitives['rotate'] = (
            CursorOrthogonalCameraSpaceRotationComponent(
                scene_component,
                components['cursor'],
                check_collision=config.check_collision,
                truncate_on_failure=config.truncate_on_failure,
        ))
        #action_primitives['rotate'] = (
        #    CursorRotateSnapAboutAxisComponent(
        #        scene_component,
        #        components['cursor'],
        #        check_collision=config.check_collision,
        #))
    
    if config.include_translate:
        action_primitives['translate'] = (
            CursorOrthogonalCameraSpaceTranslateComponent(
                scene_component,
                components['cursor'],
                check_collision=config.check_collision,
                truncate_on_failure=config.truncate_on_failure,
        ))
    
    # removal
    if config.include_remove:
        action_primitives['remove'] = CursorRemoveBrickComponent(
            scene_component,
            components['cursor'],
            check_collision=config.check_collision,
            truncate_on_failure=config.truncate_on_failure,
        )
    
    # insert
    if config.include_insert:
        action_primitives['insert'] = InsertBrickComponent(scene_component)
    
    # done
    if config.include_done:
        action_primitives['done'] = DoneComponent()
    
    # phase
    if config.include_phase:
        action_primitives['phase'] = BreakAndMakePhaseSwitchComponent(
            scene_component)
    
    # assemble step
    if config.include_assemble_step:
        assert config.include_phase
        action_primitives['assemble_step'] = AssembleStepComponent(
            action_primitives['phase'])
    
    # make the mode switch
    components['action_primitives'] = SuperMechaComponentSwitch(
        action_primitives, switch_name='mode')
    
    components['pos_snap_render'] = pos_snap_render_component
    components['neg_snap_render'] = neg_snap_render_component
    
    return components
