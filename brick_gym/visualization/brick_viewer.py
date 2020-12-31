import time
import sys
import os

import numpy

import PIL.Image as Image

import renderpy.buffer_manager_glut as buffer_manager
from renderpy.frame_buffer import FrameBufferWrapper
import renderpy.core as core
import renderpy.camera as camera
from renderpy.interactive_camera import InteractiveCamera
import renderpy.masks as masks
import renderpy.assets as drpy_assets

import brick_gym.config as config
import brick_gym.ldraw.ldraw_renderpy as ldraw_renderpy

default_image_light = 'grey_cube'

def start_viewer(
        file_path,
        width = 512,
        height = 512,
        poll_frequency = 1024):
    
    manager = buffer_manager.initialize_shared_buffer_manager(width, height)
    config_paths = '%s:%s'%(
                    config.paths['renderpy_assets_cfg'],
                    drpy_assets.default_assets_path)
    renderer = core.Renderpy(config_paths)
    manager.show_window()
    manager.enable_window()
    
    part_mask_frame = FrameBufferWrapper(width, height, anti_alias=False)
    
    camera_control = InteractiveCamera(manager, renderer)
    
    state = {
        'render_mode' : 'color',
        'steps' : 0,
        'recent_file_change_time' : -1,
        'part_mask' : None
    }
    
    def reload_scene():
        while True:
            try:
                change_time = os.stat(file_path).st_mtime
                if change_time != state['recent_file_change_time']:
                    camera_pose = renderer.get_camera_pose()
                    
                    path, ext = os.path.splitext(file_path)
                    if ext == '.json':
                        scene = file_path
                    elif ext in ('.ldr', '.mpd'):
                        scene = ldraw_renderpy.mpd_to_renderpy(
                                open(file_path),
                                image_light_directory = default_image_light)
                    
                    renderer.load_scene(scene, clear_scene=True)
                    if state['recent_file_change_time'] != -1:
                        renderer.set_camera_pose(camera_pose)
                    state['recent_file_change_time'] = change_time
                    print('Loaded: %s'%file_path)
            except:
                print('Unable to load file: %s'%file_path)
                raise
                time.sleep(1)
                print('Retrying...')
            else:
                break
    
    def render():
        if state['steps'] % poll_frequency == 0:
            reload_scene()
        state['steps'] += 1
        
        #manager.enable_frame('part_mask')
        part_mask_frame.enable()
        renderer.mask_render(flip_y=True)
        #state['part_mask'] = manager.read_pixels('part_mask')
        state['part_mask'] = part_mask_frame.read_pixels()
        
        manager.enable_window()
        if state['render_mode'] == 'color':
            renderer.color_render(flip_y=False)
        elif state['render_mode'] == 'mask':
            renderer.mask_render(flip_y=False)
    
    def get_instance_at_location(x, y):
        color = tuple(state['part_mask'][y,x])
        if color == (0,0,0):
            return None
        instance_id = masks.color_byte_to_index(color)
        return instance_id
    
    def keypress(key, x, y):
        if key == b'm':
            if state['render_mode'] == 'color':
                state['render_mode'] = 'mask'
            else:
                state['render_mode'] = 'color'
        
        elif key == b'i':
            instance_id = get_instance_at_location(x, y)
            print('----')
            if instance_id is None:
                print('No Part Selected')
            else:
                print('Instance ID: %i'%instance_id)
                instance_data = renderer.scene_description['instances']
                instance_name = 'instance_%i'%instance_id
                mesh_name = instance_data[instance_name]['mesh_name']
                part_id = mesh_name.split('_')[-1]
                print('Part ID: %s'%part_id)
                print('Translation: %f, %f, %f'%(
                        instance_data[instance_name]['transform'][0,3],
                        instance_data[instance_name]['transform'][1,3],
                        instance_data[instance_name]['transform'][2,3]))
        
        elif key == b'h':
            instance_id = get_instance_at_location(x, y)
            print('----')
            if instance_id is None:
                print('No Part Selected')
            else:
                instance_data = renderer.scene_description['instances']
                instance_name = 'instance_%i'%instance_id
                instance_data[instance_name]['hidden'] = True
                print('Hiding part %i'%instance_id)
        
        elif key == b'v':
            instances_data = renderer.scene_description['instances']
            for instance_name, instance_data in instances_data.items():
                instance_data['hidden'] = False
        
        elif key == b's':
            pixels = manager.read_pixels(frame=None)
            image_path = './brick_viewer_%06i.png'%state['steps']
            print('Saving image to: %s'%image_path)
            Image.fromarray(numpy.flip(pixels, axis=0)).save(image_path)
    
    manager.start_main_loop(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutKeyboardFunc = keypress,
            glutMouseFunc = camera_control.mouse_button,
            glutMotionFunc = camera_control.mouse_move)
