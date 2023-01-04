from steadfast.gym.component_env import ComponentEnv

class FloatingBrick(EnvComponent):
    def __init__(self):
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
