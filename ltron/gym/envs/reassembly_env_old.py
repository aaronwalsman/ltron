import math
from collections import OrderedDict


from OpenGL import GL

import numpy

import gym
import gym.spaces as spaces
import numpy

import splendor.contexts.glut as glut

from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.render import (
        ColorRenderComponent, SegmentationRenderComponent, SnapRenderComponent)
from ltron.gym.components.disassembly import PixelDisassemblyComponent
from ltron.gym.components.rotation import RotationAroundSnap
from ltron.gym.components.pick_and_place import (
        PickAndPlace, HandspacePickAndPlace)
from ltron.gym.components.brick_inserter import HandspaceBrickInserter
from ltron.gym.components.viewpoint import (
        ControlledAzimuthalViewpointComponent)
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.reassembly import Reassembly


def reassembly_template_action():
    return {
        'viewpoint' : 0,
        
        'disassembly' : {
            'activate':False,
            'polarity':0,
            'direction':0,
            'pick':(0,0),
        },
        
        'rotate' : {
            'activate':False,
            'polarity':0,
            'direction':0,
            'pick':(0,0),
        },
        
        'pick_and_place' : {
            'activate':False,
            'polarity':0,
            'direction':0,
            'pick':(0,0),
            'place':(0,0),
        },
        
        'reassembly' : {
            'start':False,
        },
    }

def handspace_reassembly_template_action():
    return {
        'workspace_viewpoint' : 0,
        
        'handspace_viewpoint' : 0,
        
        'disassembly' : {
            'activate':False,
            'polarity':0,
            'direction':0,
            'pick':numpy.array((0,0), dtype=numpy.long),
        },
        
        'rotate' : {
            'activate':False,
            'polarity':0,
            'direction':0,
            'pick':numpy.array((0,0), dtype=numpy.long),
        },
        
        'pick_and_place' : {
            'activate':False,
            'polarity':0,
            'direction':0,
            'pick':numpy.array((0,0), dtype=numpy.long),
            'place':numpy.array((0,0), dtype=numpy.long),
            'place_at_origin':False,
        },
        
        'insert_brick' : {
            'shape_id' : 0,
            'color_id' : 0,
        },
        
        'reassembly' : {
            'start':False,
            'end':False,
        },
    }

class HandspaceReassemblyWrapper(gym.Env):
    def __init__(self, env = None, *args, **kwargs):
        if env is None:
            self.env = handspace_reassembly_env(*args, **kwargs)
        else:
            self.env = env

        # sep action space
        workspace = self.env.components['workspace_color_render']
        hand = self.env.components['handspace_color_render']
        work_width = workspace.width
        work_height = workspace.height
        hand_width = hand.width
        hand_height = hand.height

        work_snap = self.env.components['workspace_pos_snap_render']
        hand_snap = self.env.components['handspace_pos_snap_render']
        work_snap_width = work_snap.width
        work_snap_height = work_snap.height
        hand_snap_width = hand_snap.width
        hand_snap_height = hand_snap.height

        num_nodes = (
            6 + # workspace camera motion
            6 + # handspace camera motion
            1 + # disassembly
            1 + # rotate
            1 + # pick and place
            1 + # brick insertion
            1   # start reassembly
        )

        insertion = self.env.components['insert_brick']
        max_id = max(insertion.id_to_brick_shape.keys())+1
        max_color = len(insertion.colors)

        self.action_space = spaces.MultiDiscrete(
            [num_nodes, 2, 2, work_snap_height, work_snap_width, hand_snap_height,
            hand_snap_width, work_snap_height, work_snap_width, 2, max_id, max_color]
        )

        self.observation_space = spaces.Dict({
            'workspace' : spaces.Box(low=0, high=255, shape=(work_height, work_width, 3), dtype=numpy.uint8),
            'handspace' : spaces.Box(low=0, high=255, shape=(hand_height, hand_width, 3), dtype=numpy.uint8)
        })

    def reset(self):
        observation = self.env.reset()
        observation = self.convert_observation(observation)
        return observation

    def step(self, action):
        action = self.convert_action(action)
        observation, reward, terminal, info = self.env.step(action)
        observation = self.convert_observation(observation)
        return observation, reward, terminal, info

    def convert_observation(self, obs):
        return obs['workspace_color_render'], obs['handspace_color_render']

    def convert_action(self, action):
        mode, polarity, direction, pick_work_y, pick_work_x, \
        pick_hand_y, pick_hand_x, place_y, place_x, place_origin, classid, color = action
        dict_action = {}

        # viewpoint
        if mode < 6:
            workspace_viewpoint_action = mode + 1
            handspace_viewpoint_action = 0
        elif mode < 12:
            workspace_viewpoint_action = 0
            handspace_viewpoint_action = mode-6+1
        else:
            workspace_viewpoint_action = 0
            handspace_viewpoint_action = 0
        dict_action['workspace_viewpoint'] = workspace_viewpoint_action
        dict_action['handspace_viewpoint'] = handspace_viewpoint_action

        # disassembly
        dict_action['disassembly'] = {
            'activate': (mode == 12),
            'polarity': polarity,
            'direction': direction,
            'pick': (pick_work_y, pick_work_x),
        }

        # rotate
        dict_action['rotate'] = {
            'activate': (mode == 13),
            'polarity': polarity,
            'direction': direction,
            'pick': (pick_work_y, pick_work_x),
        }

        # pick and place
        dict_action['pick_and_place'] = {
            'activate': (mode == 14),
            'polarity': polarity,
            'direction': direction,
            'pick': (pick_hand_y, pick_hand_x),
            'place': (place_y, place_x),
            'place_at_origin' : place_origin,
        }

        # brick insertion
        if mode != 15: classid = -1
        dict_action['insert_brick'] = {
            'shape_id': classid,
            'color': color,
        }

        # reassembly
        dict_action['reassembly'] = {
            'start': (mode == 16),
        }

        return dict_action

class SimplifiedReassemblyWrapper(gym.Env):
    def __init__(self, *args, **kwargs):
        self.env = reassembly_env(*args, **kwargs)
        
        # setup action space
        render_component = self.env.components['color_render']
        height = render_component.height
        width = render_component.width
        num_modes = (
            6 + # camera motion
            1 + # disassembly
            1 + # rotate
            1 + # pick and place
            1   # start disassembly
        )

        snap_component = self.env.components['pos_snap_render']
        snap_height = snap_component.height
        snap_width = snap_component.width
        self.action_space = spaces.MultiDiscrete(
            num_modes, 2, 2, snap_height, snap_width, snap_height, snap_width)
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=numpy.uint8)
    
    def reset(self):
        observation = self.env.reset()
        observation = self.convert_observation(observation)
        return observation
    
    def step(self, action):
        action = self.convert_action(action)
        observation, reward, terminal, info = self.env.step(action)
        observation = self.convert_observation(observation)
        return observation, reward, terminal, info
    
    def convert_observation(self, observation):
        return observation['color_render']
    
    def convert_action(self, action):
        mode, polarity, direction, pick_y, pick_x, place_y, place_x = action
        dict_action = {}
        
        # viewpoint
        if mode < 6:
            viewpoint_action = mode + 1
        else:
            viewpoint_action = 0
        dict_action['viewpoint'] = viewpoint_action
        
        # disassembly
        dict_action['disassembly'] = {
            'activate':(mode == 6),
            'polarity':polarity,
            'direction':direction,
            'pick':numpy.array(pick_y, pick_x),
        }
        
        # rotate
        dict_action['rotate'] = {
            'activate':(mode == 7),
            'polarity':polarity,
            'direction':direction,
            'pick':(pick_y, pick_x),
        }
        
        # pick and place
        dict_action['pick_and_place'] = {
            'activate':(mode == 8),
            'polarity':polarity,
            'direction':direction,
            'pick':(pick_y, pick_x),
            'place':(place_y, place_x),
        }
        
        # reassembly
        dict_action['reassembly'] = {
            'start':(mode == 9),
        }
        
        return dict_action

class InteractiveReassemblyEnv:
    def __init__(self, **kwargs):
        width = kwargs.get('width', 256)
        height = kwargs.get('height', 256)
        render_args = {
            'opengl_mode':'glut',
            'window_width':width,
            'window_height':height,
            'load_scene':'grey_cube',
        }
        self.env = reassembly_env(render_args=render_args, **kwargs)
        
        self.scene = self.env.components['scene'].brick_scene
        self.window = self.scene.render_environment.window
        self.height = self.env.components['color_render'].height
        self.width = self.env.components['color_render'].width
        self.map_height = self.env.components['pos_snap_render'].height
        self.map_width = self.env.components['pos_snap_render'].width
        assert self.height % self.map_height == 0
        assert self.width % self.map_width == 0
        self.height_scale = self.height // self.map_height
        self.width_scale = self.width // self.map_width
        
        self.window.register_callbacks(
            glutDisplayFunc = self.render,
            glutIdleFunc = self.render,
            glutKeyboardFunc = self.key_press,
            glutKeyboardUpFunc = self.key_release,
            glutSpecialFunc = self.special_key,
        )
        
        self.polarity = '+'
        self.direction = 'pull'
        self.render_mode = 'color'
        self.pick = (0,0)
    
    def render(self):
        self.window.enable_window()
        if self.render_mode == 'color':
            self.scene.color_render(flip_y=False)
        elif self.render_mode == 'mask':
            self.scene.mask_render(flip_y=False)
        elif self.render_mode == 'snap':
            snap_instances = self.scene.get_matching_snaps(
                polarity=self.polarity)
            self.scene.snap_render_instance_id(snap_instances, flip_y=False)
    
    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        print('Reward: %f'%reward)
    
    def key_press(self, key, x, y):
        if key == b'r':
            observation = self.env.reset()
        
        elif key == b'd':
            print('Disassemble: %i, %i'%(x,y))
            xx = x // self.width_scale
            yy = y // self.height_scale
            action = reassembly_template_action()
            action['disassembly']['activate'] = True
            action['disassembly']['polarity'] = '-+'.index(self.polarity)
            action['disassembly']['direction'] = (
                ('pull', 'push').index(self.direction))
            action['disassembly']['pick'] = (yy,xx)
            self.step(action)
        
        elif key == b'p':
            print('Pick: %i, %i'%(x,y))
            xx = x // self.width_scale
            yy = y // self.height_scale
            self.pick = (yy, xx)
        
        elif key == b'[':
            print('Rotate: %i, %i'%(x,y))
            xx = x // self.width_scale
            yy = y // self.height_scale
            action = reassembly_template_action()
            action['rotate'] = {
                'activate':True,
                'polarity': '-+'.index(self.polarity),
                'direction':0,
                'pick':(yy,xx),
            }
            self.step(action)
        
        elif key == b']':
            print('Rotate: %i, %i'%(x,y))
            xx = x // self.width_scale
            yy = y // self.height_scale
            action = reassembly_template_action()
            action['rotate'] = {
                'activate':True,
                'polarity': '-+'.index(self.polarity),
                'direction':1,
                'pick':(yy,xx),
            }
            self.step(action)
        
        elif key == b'm':
            if self.render_mode == 'mask':
                self.render_mode = 'color'
            else:
                self.render_mode = 'mask'
        
        elif key == b's':
            if self.render_mode == 'snap':
                self.render_mode = 'color'
            else:
                self.render_mode = 'snap'
        
        elif key == b'-':
            print('Polarity: "-"')
            self.polarity = '-'
        
        elif key == b'+':
            print('Polarity: "+"')
            self.polarity = '+'
        
        elif key == b'<':
            print('Direction: "pull"')
            self.direction = 'pull'
        
        elif key == b'>':
            print('Direction: "push"')
            self.direction = 'push'
        
        elif key == b'|':
            if not self.env.components['reassembly'].reassembling:
                print('Switching to Reassembly')
                action = reassembly_template_action()
                action['reassembly']['start'] = 1
                self.step(action)
            else:
                print('Already Reassembling')
    
    def key_release(self, key, x, y):
        if key == b'p':
            print('Place: %i, %i'%(x,y))
            pick_y, pick_x = self.pick
            place_x = x // self.width_scale
            place_y = y // self.height_scale
            action = reassembly_template_action()
            action['pick_and_place'] = {
                'activate':True,
                'polarity':'-+'.index(self.polarity),
                'direction':('pull', 'push').index(self.direction),
                'pick':(pick_y, pick_x),
                'place':(place_y, place_x),
            }
            self.step(action)
    
    def special_key(self, key, x, y):
        if key == glut.GLUT.GLUT_KEY_LEFT:
            print('Camera Left')
            action = reassembly_template_action()
            action['viewpoint'] = 1
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_RIGHT:
            print('Camera Right')
            action = reassembly_template_action()
            action['viewpoint'] = 2
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_UP:
            print('Camera Up')
            action = reassembly_template_action()
            action['viewpoint'] = 3
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_DOWN:
            print('Camera Down')
            action = reassembly_template_action()
            action['viewpoint'] = 4
            self.step(action)
    
    def get_snap_under_mouse(x, y, polarity):
        if polarity == '-':
            render_component = self.env.components['neg_snap_render']
        elif polarity == '+':
            render_component = self.env.components['pos_snap_render']
        instance_id, snap_id = render_component.observation[y, x]
        return instance_id, snap_id
    
    def start(self):
        glut.start_main_loop()


class InteractiveHandspaceReassemblyEnv:
    def __init__(self, **kwargs):
        glut.initialize()
        self.window = glut.GlutWindowWrapper(width=256+96, height=256)

        workspace_width = kwargs.get('width', 256)
        workspace_height = kwargs.get('height', 256)
        workspace_render_args = {
            'opengl_mode':'ignore',
            'load_scene':'grey_cube',
        }
        
        handspace_width = kwargs.get('handspace_width', 96)
        handspace_height = kwargs.get('handspace_height', 96)
        handspace_render_args = {
            'opengl_mode':'ignore',
            'load_scene':'grey_cube',
        }
        
        self.env = handspace_reassembly_env(
            workspace_render_args=workspace_render_args,
            handspace_render_args=handspace_render_args,
            **kwargs,
        )

        self.workspace_scene = (
            self.env.components['workspace_scene'].brick_scene)
        self.handspace_scene = (
            self.env.components['handspace_scene'].brick_scene)
        #self.workspace_window = self.workspace_scene.render_environment.window
        #self.handspace_window = self.handspace_scene.render_environment.window
        
        self.workspace_height = (
            self.env.components['workspace_color_render'].height)
        self.workspace_width = (
            self.env.components['workspace_color_render'].width)
        self.workspace_map_height = (
            self.env.components['workspace_pos_snap_render'].height)
        self.workspace_map_width = (
            self.env.components['workspace_pos_snap_render'].width)
        assert self.workspace_height % self.workspace_map_height == 0
        assert self.workspace_width % self.workspace_map_width == 0
        self.workspace_height_scale = (
            self.workspace_height // self.workspace_map_height)
        self.workspace_width_scale = (
            self.workspace_width // self.workspace_map_width)
        
        self.handspace_height = (
            self.env.components['handspace_color_render'].height)
        self.handspace_width = (
            self.env.components['handspace_color_render'].width)
        self.handspace_map_height = (
            self.env.components['handspace_pos_snap_render'].height)
        self.handspace_map_width = (
            self.env.components['handspace_pos_snap_render'].width)
        assert self.handspace_height % self.handspace_map_height == 0
        assert self.handspace_width % self.handspace_map_width == 0
        self.handspace_height_scale = (
            self.handspace_height // self.handspace_map_height)
        self.handspace_width_scale = (
            self.handspace_width // self.handspace_map_width)
        
        self.window.register_callbacks(
            glutDisplayFunc = self.render,
            glutIdleFunc = self.render,
            glutKeyboardFunc = self.key_press,
            glutKeyboardUpFunc = self.key_release,
            glutSpecialFunc = self.special_key,
        )
        
        self.polarity = '+'
        self.direction = 'pull'
        self.render_mode = 'color'
        self.pick = (0,0)
        self.insert_shape_id = ''
        self.insert_color_id = 0
        
        self.color_ids = self.env.components['insert_brick'].color_name_to_id
        self.num_colors = len(self.color_ids)
    
    def workspace_viewport(self):
        GL.glViewport(0,0,self.workspace_width, self.workspace_height)
        GL.glScissor(0,0,self.workspace_width, self.workspace_height)
    
    def handspace_viewport(self):
        GL.glViewport(256,0,self.handspace_width, self.handspace_height)
        GL.glScissor(256,0,self.handspace_width, self.handspace_height)
    
    def render(self):
        self.window.enable_window()
        if self.render_mode == 'color':
            self.workspace_viewport()
            self.workspace_scene.color_render(flip_y=False)
            self.handspace_viewport()
            self.handspace_scene.color_render(flip_y=False)
        elif self.render_mode == 'mask':
            self.workspace_viewport()
            self.workspace_scene.mask_render(flip_y=False)
            self.handspace_viewport()
            self.handspace_scene.mask_render(flip_y=False)
        elif self.render_mode == 'snap':
            snap_instances = self.workspace_scene.get_matching_snaps(
                polarity=self.polarity)
            self.workspace_viewport()
            self.workspace_scene.snap_render_instance_id(
                snap_instances, flip_y=False)
            
            snap_instances = self.handspace_scene.get_matching_snaps(
                polarity=self.polarity)
            self.handspace_viewport()
            self.handspace_scene.snap_render_instance_id(
                snap_instances, flip_y=False)
            
    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        print('Reward: %f'%reward)
    
    def key_press(self, key, x, y):
        if x < 256:
            space = 'workspace'
        else:
            space = 'handspace'
            x = x-256
            y = y-256+96
        
        if key == b'r':
            observation = self.env.reset()
        
        elif key == b'd' and space == 'workspace':
            print('Disassemble: %i, %i'%(x,y))
            xx = x // self.workspace_width_scale
            yy = y // self.workspace_height_scale
            action = handspace_reassembly_template_action()
            action['disassembly']['activate'] = True
            action['disassembly']['polarity'] = '-+'.index(self.polarity)
            action['disassembly']['direction'] = (
                ('pull', 'push').index(self.direction))
            action['disassembly']['pick'] = (yy,xx)
            self.step(action)
        
        elif key == b'p':
            print('Pick: %s, %i, %i'%(space,x,y))
            if space == 'workspace':
                xx = x // self.workspace_width_scale
                yy = y // self.workspace_height_scale
            elif space == 'handspace':
                xx = x // self.handspace_width_scale
                yy = y // self.handspace_height_scale
            self.pick = (space, yy, xx)
        
        elif key == b'P' and space == 'handspace':
            print('Pick: %s, %i, %i'%(space,x,y))
            xx = x // self.handspace_width_scale
            yy = y // self.handspace_width_scale
            action = handspace_reassembly_template_action()
            action['pick_and_place'] = {
                'activate':True,
                'polarity':'-+'.index(self.polarity),
                'pick':(yy,xx),
                'place':(0,0),
                'place_at_origin':True,
            }

            self.step(action)
        
        elif key == b'[' and space == 'workspace':
            print('Rotate: %i, %i'%(x,y))
            xx = x // self.workspace_width_scale
            yy = y // self.workspace_height_scale
            action = handspace_reassembly_template_action()
            action['rotate'] = {
                'activate':True,
                'polarity': '-+'.index(self.polarity),
                'direction':0,
                'pick':(yy,xx),
            }
            self.step(action)
        
        elif key == b']':
            print('Rotate: %i, %i'%(x,y))
            xx = x // self.workspace_width_scale
            yy = y // self.workspace_height_scale
            action = handspace_reassembly_template_action()
            action['rotate'] = {
                'activate':True,
                'polarity': '-+'.index(self.polarity),
                'direction':1,
                'pick':(yy,xx),
            }
            self.step(action)
        
        elif key == b'm':
            if self.render_mode == 'mask':
                self.render_mode = 'color'
            else:
                self.render_mode = 'mask'
        
        elif key == b's':
            if self.render_mode == 'snap':
                self.render_mode = 'color'
            else:
                self.render_mode = 'snap'
        
        elif key == b'-':
            print('Polarity: "-"')
            self.polarity = '-'
        
        elif key == b'+':
            print('Polarity: "+"')
            self.polarity = '+'
        
        elif key == b'<':
            print('Direction: "pull"')
            self.direction = 'pull'
        
        elif key == b'>':
            print('Direction: "push"')
            self.direction = 'push'
        
        elif key == b',':
            self.insert_color_id = (self.insert_color_id - 1) % self.num_colors
            print('Color: %i'%self.insert_color_id)
        
        elif key == b'.':
            self.insert_color_id = (self.insert_color_id + 1) % self.num_colors
            print('Color: %i'%self.insert_color_id)
        
        elif key in b'0123456789':
            self.insert_shape_id += key.decode("utf-8")
            print('Class ID: %s'%self.insert_shape_id)
        
        elif key == b'\x08':
            self.insert_shape_id = self.insert_shape_id[:-1]
            print('Class ID: %s'%self.insert_shape_id)
        
        elif key == b'\r':
            try:
                insert_shape_id = int(self.insert_shape_id)
            except ValueError:
                insert_shape_id = 0
            try:
                insert_color_id = int(self.insert_color_id)
            except ValueError:
                insert_color_id = 0
            action = handspace_reassembly_template_action()
            action['insert_brick'] = {
                'shape_id':insert_shape_id,
                'color_id':insert_color_id,
            }
            self.step(action)
            self.insert_shape_id = ''
        
        elif key == b'|':
            if not self.env.components['reassembly'].reassembling:
                print('Switching to Reassembly')
                action = handspace_reassembly_template_action()
                action['reassembly']['start'] = 1
                self.step(action)
            else:
                print('Already Reassembling')
    
    def key_release(self, key, x, y):
        if x < 255:
            space = 'workspace'
        else:
            space = 'handspace'
        if key == b'p' and space == 'workspace':
            print('Place: %s, %i, %i'%(space, x, y))
            workspace, pick_y, pick_x = self.pick
            place_x = x // self.workspace_width_scale
            place_y = y // self.workspace_height_scale
            action = handspace_reassembly_template_action()
            action['pick_and_place'] = {
                'activate':True,
                'polarity':'-+'.index(self.polarity),
                #'direction':('pull', 'push').index(self.direction),
                'pick':(pick_y, pick_x),
                'place':(place_y, place_x),
                'place_at_origin':False
            }
            self.step(action)
    
    def special_key(self, key, x, y):
        if x < 256:
            viewpoint = 'workspace_viewpoint'
        else:
            viewpoint = 'handspace_viewpoint'
        if key == glut.GLUT.GLUT_KEY_LEFT:
            print('Camera Left')
            action = handspace_reassembly_template_action()
            action[viewpoint] = 1
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_RIGHT:
            print('Camera Right')
            action = handspace_reassembly_template_action()
            action[viewpoint] = 2
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_UP:
            print('Camera Up')
            action = handspace_reassembly_template_action()
            action[viewpoint] = 3
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_DOWN:
            print('Camera Down')
            action = handspace_reassembly_template_action()
            action[viewpoint] = 4
            self.step(action)
    
    def start(self):
        glut.start_main_loop()

def reassembly_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
    image_width=256,
    image_height=256,
    map_width=64,
    map_height=64,
    dataset_reset_mode='uniform',
    render_args=None,
    randomize_viewpoint=True,
    randomize_colors=True,
    check_collision=True,
    print_traceback=True,
    train=False,
):
    components = OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
        dataset, split, subset, rank, size, reset_mode=dataset_reset_mode)
    dataset_info = components['dataset'].dataset_info
    max_instances = dataset_info['max_instances_per_scene']
    
    # scene
    components['scene'] = SceneComponent(
        dataset_component=components['dataset'],
        path_location=['mpd'],
        render_args=render_args,
        track_snaps=True,
        collision_checker=check_collision,
    )
    
    # viewpoint
    if randomize_viewpoint:
        start_position='uniform'
    else:
        start_position=(0,0,0)
    components['viewpoint'] = ControlledAzimuthalViewpointComponent(
        components['scene'],
        azimuth_steps=8,
        elevation_range=[math.radians(-30), math.radians(30)],
        elevation_steps=2,
        distance_range=[250, 250],
        distance_steps=1,
        azimuth_offset=math.radians(45.),
        aspect_ratio=image_width/image_height,
        start_position=start_position,
        frame_scene=True,
    )
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponent(
            dataset_info['color_ids'],
            components['scene'],
            randomize_frequency='reset',
        )
    
    # utility rendering components
    pos_snap_render = SnapRenderComponent(
        map_width, map_height, components['scene'], polarity='+')
    neg_snap_render = SnapRenderComponent(
        map_width, map_height, components['scene'], polarity='-')
    
    # action spaces
    components['disassembly'] = PixelDisassemblyComponent(
        max_instances,
        components['scene'],
        pos_snap_render,
        neg_snap_render,
        check_collision=check_collision,
    )
    components['rotate'] = RotationAroundSnap(
        components['scene'],
        pos_snap_render,
        neg_snap_render,
        check_collision=check_collision,
    )
    components['pick_and_place'] = PickAndPlace(
        components['scene'],
        pos_snap_render,
        neg_snap_render,
        check_collision=check_collision,
    )
    
    # reassembly
    components['reassembly'] = Reassembly(
        components['scene'],
        reassembly_mode='square',
    )
    
    # color render
    components['color_render'] = ColorRenderComponent(
        image_width, image_height, components['scene'], anti_alias=True)
    
    # snap render
    components['pos_snap_render'] = pos_snap_render
    components['neg_snap_render'] = neg_snap_render
    
    # build the env
    env = LtronEnv(components, print_traceback=print_traceback)
    
    return env


def handspace_reassembly_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
    workspace_image_width=256,
    workspace_image_height=256,
    handspace_image_width=96,
    handspace_image_height=96,
    workspace_map_width=64,
    workspace_map_height=64,
    handspace_map_width=24,
    handspace_map_height=24,
    dataset_reset_mode='uniform',
    max_episode_length=32,
    workspace_render_args=None,
    handspace_render_args=None,
    randomize_viewpoint=True,
    randomize_colors=True,
    check_collision=True,
    print_traceback=True,
    train=False,
):
    components = OrderedDict()
    
    # dataset
    components['dataset'] = DatasetPathComponent(
        dataset, split, subset, rank, size, reset_mode=dataset_reset_mode)
    dataset_info = components['dataset'].dataset_info
    shape_ids = dataset_info['shape_ids']
    color_ids = dataset_info['color_ids']
    max_instances = dataset_info['max_instances_per_scene']
    max_edges = dataset_info['max_edges_per_scene']
    
    # scenes
    components['workspace_scene'] = SceneComponent(
        dataset_component=components['dataset'],
        path_location=['mpd'],
        render_args=workspace_render_args,
        track_snaps=True,
        collision_checker=check_collision,
    )
    
    components['handspace_scene'] = SceneComponent(
        render_args=handspace_render_args,
        track_snaps=True,
        collision_checker=False,
    )
    
    components['episode'] = MaxEpisodeLengthComponent(
        max_episode_length, observe_step=False)
    
    # viewpoint
    azimuth_steps = 8
    elevation_range = [math.radians(-30), math.radians(30)]
    elevation_steps = 2
    distance_steps = 1
    if randomize_viewpoint:
        start_position='uniform'
    else:
        start_position=(0,0,0)
    components['workspace_viewpoint'] = ControlledAzimuthalViewpointComponent(
        components['workspace_scene'],
        azimuth_steps=azimuth_steps,
        elevation_range=elevation_range,
        elevation_steps=elevation_steps,
        distance_range=[250,250],
        distance_steps=distance_steps,
        aspect_ratio=workspace_image_width/workspace_image_height,
        start_position=start_position,
        frame_scene=True,
    )
    
    components['handspace_viewpoint'] = ControlledAzimuthalViewpointComponent(
        components['handspace_scene'],
        azimuth_steps=azimuth_steps,
        elevation_range=elevation_range,
        elevation_steps=elevation_steps,
        distance_range=[150,150],
        distance_steps=distance_steps,
        aspect_ratio=handspace_image_width/handspace_image_height,
        start_position=(0,0,0),
        frame_scene=True,
    )
    
    # color randomization
    if randomize_colors:
        components['color_randomization'] = RandomizeColorsComponent(
            dataset_info['color_ids'],
            components['workspace_scene'],
            randomize_frequency='reset',
        )
    
    # utility rendering components
    workspace_pos_snap_render = SnapRenderComponent(
        workspace_map_width,
        workspace_map_height,
        components['workspace_scene'],
        polarity='+',
    )
    workspace_neg_snap_render = SnapRenderComponent(
        workspace_map_width,
        workspace_map_height,
        components['workspace_scene'],
        polarity='-',
    )
    
    handspace_pos_snap_render = SnapRenderComponent(
        handspace_map_width,
        handspace_map_height,
        components['handspace_scene'],
        polarity='+',
    )
    handspace_neg_snap_render = SnapRenderComponent(
        handspace_map_width,
        handspace_map_height,
        components['handspace_scene'],
        polarity='-',
    )
    
    # action spaces
    components['disassembly'] = PixelDisassemblyComponent(
        max_instances,
        components['workspace_scene'],
        workspace_pos_snap_render,
        workspace_neg_snap_render,
        handspace_component=components['handspace_scene'],
        check_collision=check_collision,
    )
    components['rotate'] = RotationAroundSnap(
        components['workspace_scene'],
        workspace_pos_snap_render,
        workspace_neg_snap_render,
        check_collision=check_collision,
    )
    components['pick_and_place'] = HandspacePickAndPlace(
        components['workspace_scene'],
        workspace_pos_snap_render,
        workspace_neg_snap_render,
        components['handspace_scene'],
        handspace_pos_snap_render,
        handspace_neg_snap_render,
        check_collision=check_collision,
    )
    components['insert_brick'] = HandspaceBrickInserter(
        components['handspace_scene'],
        components['workspace_scene'],
        shape_ids,
        color_ids,
        max_instances,
    )
    
    # reassembly
    components['reassembly'] = Reassembly(
        shape_ids=shape_ids,
        color_ids=color_ids,
        max_instances=max_instances,
        max_edges=max_edges,
        workspace_scene_component=components['workspace_scene'],
        workspace_viewpoint_component=components['workspace_viewpoint'],
        handspace_scene_component=components['handspace_scene'],
        dataset_component=components['dataset'],
        reassembly_mode='clear',
        train=train,
    )
    
    # color render
    components['workspace_color_render'] = ColorRenderComponent(
        workspace_image_width,
        workspace_image_height,
        components['workspace_scene'],
        anti_alias=True,
    )
    
    components['handspace_color_render'] = ColorRenderComponent(
        handspace_image_width,
        handspace_image_height,
        components['handspace_scene'],
        anti_alias=True,
    )
    
    if train:
        components['workspace_segmentation_render'] = (
             SegmentationRenderComponent(
                workspace_map_width,
                workspace_map_height,
                components['workspace_scene'],
            )
        )
    
    # snap render
    components['workspace_pos_snap_render'] = workspace_pos_snap_render
    components['workspace_neg_snap_render'] = workspace_neg_snap_render
    components['handspace_pos_snap_render'] = handspace_pos_snap_render
    components['handspace_neg_snap_render'] = handspace_neg_snap_render
    
    # build the env
    env = LtronEnv(components, print_traceback=print_traceback)
    
    return env

if __name__ == '__main__':
    #interactive_env = InteractiveReassemblyEnv(
    interactive_env = InteractiveHandspaceReassemblyEnv(
        dataset='random_six',
        split='simple_single',
        subset=1,
        train=True,
        randomize_colors=False,
        randomize_viewpoint=False)
    interactive_env.start()
