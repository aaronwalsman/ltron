#!/usr/bin/env python
import math
from collections import OrderedDict


from OpenGL import GL

import numpy

import gym
import gym.spaces as spaces
import numpy

import splendor.contexts.glut as glut

from ltron.gym.envs.ltron_env import LtronEnv
from ltron.gym.envs.break_and_make_env import (
    break_and_make_env)

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
        
        self.env = break_and_make_env(
            workspace_render_args=workspace_render_args,
            handspace_render_args=handspace_render_args,
            **kwargs,
        )

        self.workspace_scene = (
            self.env.components['workspace_scene'].brick_scene)
        self.handspace_scene = (
            self.env.components['handspace_scene'].brick_scene)
        
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
            glutMouseFunc = self.mouse_button,
        )
        
        self.render_mode = 'color'
        self.insert_class_id = ''
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
            '''
            snap_instances = self.workspace_scene.get_matching_snaps(
                polarity=self.polarity)
            self.workspace_viewport()
            self.workspace_scene.snap_render_instance_id(
                snap_instance, flip_y=False)
            
            snap_instances = self.handspace_scene.get_matching_snaps(
                polarity=self.polarity)
            self.handspace_viewport()
            self.handspace_scene.snap_render_instance_id(
                snap_instances, flip_y=False)
            '''
            
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
            action = self.env.no_op_action()
            action['disassembly'] = 1
            self.step(action)
        
        elif key == b'p':
            action = self.env.no_op_action()
            action['pick_and_place'] = 1
            self.step(action)
        
        elif key == b'P' and space == 'handspace':
            action = self.env.no_op_action()
            action['pick_and_place'] = 2
            self.step(action)
        
        elif key == b'[' and space == 'workspace':
            print('Rotate: %i, %i'%(x,y))
            action = self.env.no_op_action()
            action['rotate'] = 1
            self.step(action)
        
        elif key == b']':
            print('Rotate: %i, %i'%(x,y))
            action = self.env.no_op_action()
            action['rotate'] = 3
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
        
        elif key == b',':
            self.insert_color_id = (self.insert_color_id - 1) % self.num_colors
            print('Color: %i'%self.insert_color_id)
        
        elif key == b'.':
            self.insert_color_id = (self.insert_color_id + 1) % self.num_colors
            print('Color: %i'%self.insert_color_id)
        
        elif key in b'0123456789':
            self.insert_class_id += key.decode("utf-8")
            print('Class ID: %s'%self.insert_class_id)
        
        elif key == b'\x08':
            self.insert_class_id = self.insert_class_id[:-1]
            print('Class ID: %s'%self.insert_class_id)
        
        elif key == b'\r':
            try:
                insert_class_id = int(self.insert_class_id)
            except ValueError:
                insert_class_id = 0
            try:
                insert_color_id = int(self.insert_color_id)
            except ValueError:
                insert_color_id = 0
            action = self.env.no_op_action()
            action['insert_brick'] = {
                'class_id':insert_class_id,
                'color_id':insert_color_id,
            }
            self.step(action)
            self.insert_class_id = ''
        
        elif key == b'|':
            if not self.env.components['reassembly'].reassembling:
                print('Switching to Reassembly')
                action = self.env.no_op_action()
                action['reassembly'] = 1
                self.step(action)
            else:
                print('Already Reassembling')
    
    def mouse_button(self, button, button_state, x, y):
        if button_state != 0:
            return
        
        if button == 0:
            polarity = 0
        elif button == 2:
            polarity = 1
        else:
            return
        
        if x < 256:
            space = 'workspace'
        else:
            space = 'handspace'
            x = x-256
            y = y-256+96
        
        
        action = self.env.no_op_action()
        if space == 'workspace':
            yy = y // self.workspace_height_scale
            xx = x // self.workspace_width_scale
            action['workspace_cursor'] = {
                'activate':True,
                'position':[yy,xx],
                'polarity':polarity,
            }
            print(yy,xx)
        elif space == 'handspace':
            yy = y // self.handspace_height_scale
            xx = x // self.handspace_width_scale
            action['handspace_cursor'] = {
                'activate':True,
                'position':[yy,xx],
                'polarity':polarity,
            }
            print(yy,xx)
        self.step(action)
        
    def key_release(self, key, x, y):
        pass
        '''
        if x < 255:
            space = 'workspace'
        else:
            space = 'handspace'
        if key == b'p' and space == 'workspace':
            print('Place: %s, %i, %i'%(space, x, y))
            workspace, pick_y, pick_x = self.pick
            place_x = x // self.workspace_width_scale
            place_y = y // self.workspace_height_scale
            action = self.env.no_op_action()
            action['pick_and_place'] = {
                'activate':True,
                'polarity':'-+'.index(self.polarity),
                #'direction':('pull', 'push').index(self.direction),
                'pick':(pick_y, pick_x),
                'place':(place_y, place_x),
                'place_at_origin':False
            }
            self.step(action)
        '''
    
    def special_key(self, key, x, y):
        if x < 256:
            viewpoint = 'workspace_viewpoint'
        else:
            viewpoint = 'handspace_viewpoint'
        if key == glut.GLUT.GLUT_KEY_LEFT:
            print('Camera Left')
            action = self.env.no_op_action()
            action[viewpoint] = 1
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_RIGHT:
            print('Camera Right')
            action = self.env.no_op_action()
            action[viewpoint] = 2
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_UP:
            print('Camera Up')
            action = self.env.no_op_action()
            action[viewpoint] = 3
            self.step(action)
        elif key == glut.GLUT.GLUT_KEY_DOWN:
            print('Camera Down')
            action = self.env.no_op_action()
            action[viewpoint] = 4
            self.step(action)
    
    def start(self):
        glut.start_main_loop()

if __name__ == '__main__':
    interactive_env = InteractiveHandspaceReassemblyEnv(
        dataset='omr_split_4',
        split='tmp',
        subset=None,
        train=False,
        randomize_colors=False,
        randomize_viewpoint=False)
    interactive_env.start()
