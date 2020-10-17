#!/usr/bin/env python
import time
import sys
import os
import numpy
import renderpy.buffer_manager_glut as buffer_manager
import renderpy.core as core
import renderpy.camera as camera
import renderpy.interactive_camera as interactive_camera

import brick_gym.ldraw.colors as colors
import brick_gym.ldraw.ldraw_renderpy as ldraw_renderpy

default_image_light = '/home/awalsman/Development/renderpy/renderpy/example_image_lights/grey_cube'

def start_viewer(
        file_path,
        width = 512,
        height = 512,
        poll_frequency = 1024):
    
    manager = buffer_manager.initialize_shared_buffer_manager(width, height)
    renderer = core.Renderpy()
    manager.show_window()
    manager.enable_window()
    
    manager.add_frame('part_mask', width, height, anti_aliasing=False)
    
    camera_control = interactive_camera.InteractiveCamera(
            manager, renderer, 1, 5000)
    
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
                    
                    renderer.load_scene(scene, clear_existing=True)
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
        
        manager.enable_frame('part_mask')
        renderer.mask_render(flip_y=True)
        state['part_mask'] = manager.read_pixels('part_mask')
        
        manager.enable_window()
        if state['render_mode'] == 'color':
            renderer.color_render(flip_y=False)
        elif state['render_mode'] == 'mask':
            renderer.mask_render(flip_y=False)
    
    def get_instance_at_location(x, y):
        color = tuple(state['part_mask'][y,x])
        if color == (0,0,0):
            return None
        instance_id = colors.mask_color_indices[color]
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
    
    def mouse_button(button, button_state, x, y):
        return camera_control.mouse_button(button, button_state, x, y)
    
    def mouse_move(x, y):
        return camera_control.mouse_move(x, y)
    
    manager.start_main_loop(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutKeyboardFunc = keypress,
            glutMouseFunc = mouse_button,
            glutMotionFunc = mouse_move)

if __name__ == '__main__':
    width = 1024
    height = 1024
    
    file_path = sys.argv[1]
    
    start_viewer(
            file_path,
            width = width,
            height = height)
