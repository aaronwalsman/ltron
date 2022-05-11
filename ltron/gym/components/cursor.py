import random

import numpy

from gym.spaces import Dict, Discrete, MultiDiscrete
from ltron.gym.spaces import MultiScreenPixelSpace, SymbolicSnapSpace

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class CursorComponent(LtronGymComponent):
    def __init__(self, randomize_starting_position=True):
        self.randomize_starting_position = randomize_starting_position
    
    def reset(self):
        if self.randomize_starting_position:
            action = random.randint(1, self.action_space.n-1)
            coords = self.action_space.unravel(action)
            self.set_cursor(*coords)
        else:
            self.set_cursor(*self.zero)
        return self.observe()
    
    def step(self, action):
        if action:
            coords = self.action_space.unravel(action)
            self.set_cursor(*coords)
        observation = self.observe()
        return observation, 0, False, {}

    def set_cursor(self, *coords):
        # test to make sure coords are compatible with the action space
        a = self.action_space.ravel(*coords)
        self.coords = coords

    def observe(self):
        self.observation = self.action_space.ravel(*self.coords)

        return self.observation
    
    def get_state(self):
        return self.coords
    
    def set_state(self, coords):
        self.coords = coords

    def no_op_action(self):
        return 0

class SymbolicCursor(CursorComponent):
    def __init__(self,
        assembly_components,
        randomize_starting_position=True,
    ):
        super().__init__(
            randomize_starting_position=randomize_starting_position,
        )
        self.assembly_components = assembly_components
        self.assembly_order = list(self.assembly_components.keys())
        
        self.action_space = SymbolicSnapSpace({
            component_name:component.max_instances
            for component_name, component in assembly_components.items()
        })
        self.observation_space = self.action_space

    def get_selected_snap(self):
        name = self.coords[0]
        if name == 'NO_OP':
            return self.name, 0, 0
        i,s = self.coords[1:]
        assembly = self.assembly_components[name].observe()
        if assembly['shape'][i] == 0:
            return name, 0, 0
        else:
            return name, i, s

    def actions_to_select_snap(self, name, instance_id, snap_id):
        assembly = self.assembly_components[name].observe()
        if assembly['shape'][instance_id] == 0:
            return []
        else:
            return [self.action_space.ravel(name, instance_id, snap_id)]
    
    def visible_snaps(self, names=None):
        snaps = []
        if names is None:
            names = self.assembly_components.keys()
        for name in names:
            component = self.assembly_components[name]
            assembly = component.observe()
            scene = component.scene_component.brick_scene
            for i, shape in enumerate(assembly['shape']):
                if shape != 0:
                    instance = scene.instances[i]
                    for snap in instance.snaps:
                        snaps.append((name, i, int(snap.snap_style)))
        
        return snaps

class MultiViewCursor(CursorComponent):
    def __init__(self,
        max_instances_per_scene,
        pos_render_components,
        neg_render_components,
        randomize_starting_position=True,
    ):
        super().__init__(
            randomize_starting_position=randomize_starting_position
        )
        self.max_instances_per_scene = max_instances_per_scene
        self.pos_render_components = pos_render_components
        self.neg_render_components = neg_render_components
        self.zero = next(iter(pos_render_components.keys())), 0, 0, 0
        
        screen_dimensions = {
            n : (c.height, c.width, 2)
            for n, c in pos_render_components.items()}
        assert all(
            (c.height, c.width, 2) == screen_dimensions[n]
            for n, c in neg_render_components.items()
        )
        self.action_space = MultiScreenPixelSpace(screen_dimensions)
    
    def get_selected_snap(self):
        if self.polarity:
            render_component = self.pos_render_components[self.screen_name]
        else:
            render_component = self.neg_render_components[self.screen_name]
        
        snap_map = render_component.observe()
        instance_id, snap_id = snap_map[self.position[0], self.position[1]]
        return self.screen_name, instance_id, snap_id
    
    def actions_to_select_snap(self, screen_name, instance, snap):
        actions = []
        pos_component = self.pos_render_components[screen_name]
        neg_component = self.neg_render_components[screen_name]
        for p, component in (1, pos_component), (0, neg_component):
            snap_map = component.observe()
            ys, xs = numpy.where(
                (snap_map[:,:,0] == instance) &
                (snap_map[:,:,1] == snap)
            )
            for y, x in zip(ys, xs):
                actions.append(self.action_space.ravel(screen_name, y, x, p))
        
        return actions
    
    def visible_snaps(self, names=None):
        snaps = set()
        o = self.max_instances_per_scene + 1
        for comps in (self.pos_render_components, self.neg_render_components):
            if names is None:
                comp_names = comps.keys()
            else:
                comp_names = names
            
            for name in comp_names:
                component = comps[name]
                render = component.observe()
                render = render[...,0] + render[...,1] * o
                visible_snap_instances = numpy.unique(render)
                visible_instances = visible_snap_instances % o
                visible_snaps = visible_snap_instances // o
                snaps = snaps | set(
                    (name, i, s)
                    for i, s in zip(visible_instances, visible_snaps)
                )
        
        return list(snaps)
