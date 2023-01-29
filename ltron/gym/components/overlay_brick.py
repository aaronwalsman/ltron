import math

import numpy

from supermecha import SuperMechaComponent

# ok, this is going to have a brick instance ID, and will take that
# brick, scale it tiny, and move it close to the camera, so it acts
# like a floating overlay
# this brick needs to be excluded from all collision checks
# anything that interacted with the "hand" before now needs to
# interact with this instead

# needs a scene
# inserter interfaces with this
# pick and place interfaces with this
# disassemble interfaces with this
    # do we need this?  can this just be pick and place?
    # maybe when you pick and then place to an empty location, that
    # disassembles to the floating brick?
# needs hand viewpoint control
# this needs to operate AFTER table viewpoint to put the floating
# instance in the right place

# now there will be:
    # only one scene
    # only one set of renderers
    # cursor over only a single window (yay, glut human interface!)

# are we going to modify the actual brick scene API for this or make it
# purely a gym thing?  I think I vote gym.  Although the action things
# have been pushed up to BrickScene.

class OverlayBrickComponent(SuperMechaComponent):
    def __init__(self,
        scene_component,
        table_viewpoint_component,
        hand_viewpoint_component,
        scale = 0.1,
    ):
        self.scene_component = scene_component
        self.table_viewpoint_component = table_viewpoint_component
        self.hand_viewpoint_component = hand_viewpoint_component
        self.scale = scale
    
    def get_instance(self):
        scene = self.scene_component.brick_scene
        return scene.instances.get(self.overlay_instance, None)
    
    def place_overlay_instance(self):
        instance = self.get_instance()
        if instance is not None:
            scene = self.scene_component.brick_scene
            scale_transform = numpy.eye(4)
            scale_transform[:3,:3] *= self.scale
            #scale_transform[0,3] = -20 * math.sin(fov/4.)
            #scale_transform[1,3] = 20 * math.sin(fov/4.)
            #scale_transform[2,3] = -20
            
            table_camera_matrix = self.table_viewpoint_component.camera_matrix
            hand_view_matrix = self.hand_viewpoint_component.view_matrix
            
            fov = self.table_viewpoint_component.field_of_view
            d = hand_view_matrix[2,3]
            x_offset = d * math.sin(fov/2.) * 0.667
            y_offset = -d * math.sin(fov/2.) * 0.667
            shift_transform = numpy.eye(4)
            shift_transform[0,3] = x_offset
            shift_transform[1,3] = y_offset
            
            overlay_transform = (
                table_camera_matrix @
                scale_transform @
                shift_transform @
                hand_view_matrix @
                scene.upright
            )
            
            scene.move_instance(instance, overlay_transform)
    
    def set_overlay_instance(self, instance_id):
        
        # remove any existing instance
        instance = self.get_instance()
        if instance_id != 0 and instance is not None:
            scene = self.scene_component.brick_scene
            scene.remove_instance(instance)
        
        # update the instance id
        self.overlay_instance = int(instance_id)
        
        # move the instance into place
        self.place_overlay_instance()
    
    def is_overlaid(self, instance):
        return self.overlay_instance == int(instance)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.overlay_instance = 0
        return None, {}
    
    def step(self, action):
        self.place_overlay_instance()
        return None, 0., False, False, {}
