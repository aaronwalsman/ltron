import time
import math
import sys
import os

import numpy

import PIL.Image as Image

from pyquaternion import Quaternion

#import renderpy.buffer_manager_glut as buffer_manager
import renderpy.contexts.glut as drpy_glut
from renderpy.frame_buffer import FrameBufferWrapper
import renderpy.core as core
import renderpy.camera as camera
from renderpy.interactive_camera import InteractiveCamera
import renderpy.masks as masks
import renderpy.assets as drpy_assets
from renderpy.image import save_image

import ltron.settings as settings
from ltron.dataset.paths import resolve_subdocument
import ltron.ldraw.paths as ldraw_paths
#import ltron.ldraw.ldraw_renderpy as ldraw_renderpy
from ltron.bricks.brick_scene import BrickScene

def start_viewer(
        file_path,
        width = 512,
        height = 512,
        image_light = 'grey_cube',
        poll_frequency = 1024,
        print_fps = False):
    
    resolved_file_path, subdocument = resolve_subdocument(file_path)
    
    scene = BrickScene(
        renderable=True,
        render_args={
            'opengl_mode':'glut',
            'window_width':width,
            'window_height':height},
        track_snaps=True)
    
    window = scene.render_environment.window
    renderer = scene.render_environment.renderer
    
    '''
    scene.load_image_light(
            image_light,
            diffuse_texture=image_light + '_dif',
            reflect_texture=image_light + '_ref',
            set_active=True)
    '''
    
    scene.add_direction_light(
            'front',
            direction=(0,-0.707,-0.707),
            color=(2,2,2)
    )
    
    scene.add_direction_light(
            'back_a',
            direction=(0.5,-0.5,0.5),
            color=(1,1,1)
    )
    
    scene.add_direction_light(
            'back_b',
            direction=(-0.5,-0.5,0.5),
            color=(1,1,1)
    )
    
    scene.set_ambient_color((0.3,0.3,0.3))
    
    scene.set_background_color((0.65,0.65,0.65))
    
    window.set_active()
    window.enable_window()
    
    part_mask_frame = FrameBufferWrapper(width, height, anti_alias=False)
    
    camera_control = InteractiveCamera(window, renderer)
    
    state = {
        'render_mode' : 'color',
        'steps' : 0,
        'recent_file_change_time' : -1,
        'part_mask' : None,
        'snap_mode' : 'none',
        'batch_time' : time.time(),
        'pick_snap' : None,
    }
    
    def reload_scene(force=False):
        while True:
            try:
                change_time = os.stat(resolved_file_path).st_mtime
                if change_time != state['recent_file_change_time'] or force:
                    camera_pose = scene.get_camera_pose()
                    scene.instances.clear()
                    scene.import_ldraw(file_path)
                    
                    #renderer.load_scene(scene, clear_scene=True)
                    if state['recent_file_change_time'] == -1:
                        scene.camera_frame_scene(azimuth=-2.5, elevation=-0.3)
                    else:
                        scene.set_camera_pose(camera_pose)
                    state['recent_file_change_time'] = change_time
                    print('Loaded: %s'%file_path)
                    print('Brick Types: %i'%len(scene.brick_library))
                    print('Brick Instances: %i'%len(scene.instances))
                    print('Colors: %i'%len(scene.color_library))
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
            t_now = time.time()
            if print_fps:
                print('fps: %.04f'%(
                        poll_frequency / (t_now - state['batch_time'])))
            state['batch_time'] = t_now
        state['steps'] += 1
        
        '''
        part_mask_frame.enable()
        scene.mask_render(flip_y=True)
        state['part_mask'] = part_mask_frame.read_pixels()
        '''
        
        window.enable_window()
        if state['render_mode'] == 'color':
            scene.color_render(flip_y=False)
        if state['render_mode'] == 'removable':
            scene.removable_render(flip_y=False)
        elif state['render_mode'] == 'mask':
            scene.mask_render(flip_y=False)
    
    def get_instance_at_location(x, y):
        part_mask_frame.enable()
        scene.mask_render(flip_y=True)
        state['part_mask'] = part_mask_frame.read_pixels()
        
        color = tuple(state['part_mask'][y,x])
        if color == (0,0,0):
            return None
        instance_id = masks.color_byte_to_index(color)
        return instance_id
    
    def key_press(key, x, y):
        if key == b'm':
            if state['render_mode'] != 'mask':
                state['render_mode'] = 'mask'
            else:
                state['render_mode'] = 'color'
        
        if key == b'r':
            if state['render_mode'] != 'removable':
                state['render_mode'] = 'removable'
            else:
                state['render_mode'] = 'color'
        
        elif key == b'i':
            instance_id = get_instance_at_location(x, y)
            print('----')
            if instance_id is None:
                print('No Part Selected')
            else:
                print('Instance ID: %i'%instance_id)
                instance = scene.instances[instance_id]
                transform = instance.transform
                type_name = instance.brick_type.reference_name
                print('Part Name: %s'%type_name)
                print('Translation: %f, %f, %f'%(
                        transform[0,3],
                        transform[1,3],
                        transform[2,3]))
        
        elif key == b'h':
            instance_id = get_instance_at_location(x, y)
            print('----')
            if instance_id is None:
                print('No Part Selected')
            else:
                #instance = scene.get_instance(instance_id)
                scene.hide_instance(str(instance_id))
                print('Hiding Brick Instance %i'%instance_id)
        
        elif key == b'H':
            print('----')
            print('Hiding All Instances')
            scene.hide_all_brick_instances()
        
        elif key == b'v':
            scene.show_all_brick_instances()
            print('----')
            print('Showing All Hidden Instances')
        
        elif key == b'w':
            pixels = window.read_pixels()
            image_path = './brick_viewer_%06i.png'%state['steps']
            print('----')
            print('Writing Image To: %s'%image_path)
            Image.fromarray(numpy.flip(pixels, axis=0)).save(image_path)
        
        elif key == b'd':
            part_mask_frame.enable()
            scene.mask_render(flip_y=True)
            depth_map = part_mask_frame.read_pixels(
                    read_depth=True, projection=renderer.get_projection())
            min_depth = numpy.min(depth_map)
            max_depth = numpy.max(depth_map)
            depth_map = 1.0 - (depth_map - min_depth) / (500 - min_depth)
            depth_map = numpy.clip(depth_map, 0, 1)
            depth_map = (depth_map * 255).astype(numpy.uint8)
            print(depth_map.shape)
            
            image_path = './brick_viewer_depth_%06i.png'%state['steps']
            print('----')
            print('Writing Depth Image To: %s'%image_path)
            Image.fromarray(depth_map[...,0]).save(image_path)
        
        elif key == b's':
            if state['snap_mode'] == 'none':
                state['snap_mode'] = 'all'
                scene.show_all_snap_instances()
            else:
                state['snap_mode'] = 'none'
                scene.hide_all_snap_instances()
        
        elif key == b'c':
            instance_id = get_instance_at_location(x, y)
            connected_snaps = scene.get_instance_snap_connections(
                    str(instance_id))
            print('----')
            print('Instance: %i'%instance_id)
            print('All Snaps:')
            print(connected_snaps)
            
            connected_instances = set(snap[0] for snap in connected_snaps)
            print('All Connected Instances:')
            print(list(sorted(connected_instances)))
            
            unidirectional = set(
                    s for s in connected_instances if int(s) > instance_id)
            print('Outgoing Unidirectional Edges:')
            print(list(sorted(unidirectional)))
        
        elif key == b'p' or key == b'n':
            if key == b'p':
                # render positive snaps in mask mode
                instance_id, snap_id = get_snap_under_mouse(x, y, '+')
                print('picked: %s, %s'%(instance_id, snap_id))
                if instance_id:
                    state['pick_snap'] = (instance_id, snap_id)
            
            if key == b'n':
                # render negative snaps in mask mode
                instance_id, snap_id = get_snap_under_mouse(x, y, '-')
                print('picked: %s, %s'%(instance_id, snap_id))
                if instance_id:
                    state['pick_snap'] = (instance_id, snap_id)
        
        elif key == b',':
            instance_id, snap_id = get_snap_under_mouse(x, y, '+')
            transform = numpy.array([
                    [ 0, 0,-1, 0],
                    [ 0, 1, 0, 0],
                    [ 1, 0, 0, 0],
                    [ 0, 0, 0, 1]])
            transform_about_snap(instance_id, snap_id, transform)
        
        elif key == b'<':
            instance_id, snap_id = get_snap_under_mouse(x, y, '-')
            transform = numpy.array([
                    [ 0, 0,-1, 0],
                    [ 0, 1, 0, 0],
                    [ 1, 0, 0, 0],
                    [ 0, 0, 0, 1]])
            transform_about_snap(instance_id, snap_id, transform)
        
        elif key == b'.':
            instance_id, snap_id = get_snap_under_mouse(x, y, '+')
            transform = numpy.array([
                    [ 0, 0, 1, 0],
                    [ 0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [ 0, 0, 0, 1]])
            transform_about_snap(instance_id, snap_id, transform)
        
        elif key == b'>':
            instance_id, snap_id = get_snap_under_mouse(x, y, '-')
            transform = numpy.array([
                    [ 0, 0, 1, 0],
                    [ 0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [ 0, 0, 0, 1]])
            transform_about_snap(instance_id, snap_id, transform)
        
        elif key == b'l':
            print('reloading scene')
            reload_scene(force=True)
        
        elif key == b'e':
            print('exporting scene')
            scene.export_ldraw('./export.mpd')
        
        elif key == b'E':
            print('overwriting scene')
            scene.export_ldraw(resolved_file_path)
        
        elif key == b'[':
            instance_id = get_instance_at_location(x, y)
            transform = scene.instances[instance_id].transform
            transform[1,3] -= 8
            scene.move_instance(instance_id, transform)
        
        elif key == b']':
            instance_id = get_instance_at_location(x, y)
            transform = scene.instances[instance_id].transform
            transform[1,3] += 8
            scene.move_instance(instance_id, transform)
    
    def special_key(key, x, y):
        if key == drpy_glut.GLUT.GLUT_KEY_DOWN:
            instance_id = get_instance_at_location(x, y)
            transform = scene.instances[instance_id].transform
            transform[2,3] -= 10
            scene.move_instance(instance_id, transform)
        elif key == drpy_glut.GLUT.GLUT_KEY_UP:
            instance_id = get_instance_at_location(x, y)
            transform = scene.instances[instance_id].transform
            transform[2,3] += 10
            scene.move_instance(instance_id, transform)
        elif key == drpy_glut.GLUT.GLUT_KEY_LEFT:
            instance_id = get_instance_at_location(x, y)
            transform = scene.instances[instance_id].transform
            transform[0,3] -= 10
            scene.move_instance(instance_id, transform)
        elif key == drpy_glut.GLUT.GLUT_KEY_RIGHT:
            instance_id = get_instance_at_location(x, y)
            transform = scene.instances[instance_id].transform
            transform[0,3] += 10
            scene.move_instance(instance_id, transform)
    
    def key_release(key, x, y):
        if key == b'p' or key == b'n':
            if key == b'p':
                # render negative snaps in mask mode
                dst_instance_id, dst_snap_id = get_snap_under_mouse(x, y, '-')
                print('placing to: %s, %s'%(dst_instance_id, dst_snap_id))
                if dst_instance_id and state['pick_snap'] is not None:
                    src_instance_id, src_snap_id = state['pick_snap']
                    src_instance = scene.instances[src_instance_id]
                    src_transform = src_instance.get_snap(
                            src_snap_id).transform
                    dst_transform = scene.instances[dst_instance_id].get_snap(
                            dst_snap_id).transform
                    
                    best_transform = None
                    best_distance = -float('inf')
                    for i in range(4):
                        theta = i * math.pi/2
                        c = math.cos(theta)
                        s = math.sin(theta)
                        r = numpy.array([
                                [ c, 0, s, 0],
                                [ 0, 1, 0, 0],
                                [-s, 0, c, 0],
                                [ 0, 0, 0, 1]])
                        new_transform = (
                                dst_transform @
                                r @
                                numpy.linalg.inv(src_transform) @
                                src_instance.transform)
                        offset = (
                                new_transform @
                                numpy.linalg.inv(src_instance.transform))
                        pseudo_angle = numpy.trace(offset[:3,:3])
                        if pseudo_angle > best_distance:
                            best_transform = new_transform
                            best_distance = pseudo_angle
                    
                    scene.move_instance(
                            src_instance, best_transform)
                    
                    '''
                    offset = dst_transform @ numpy.linalg.inv(src_transform)
                    scene.move_instance(
                            src_instance, offset @ src_instance.transform)
                    '''
            
            elif key == b'n':
                # render positive snaps in mask mode
                dst_instance_id, dst_snap_id = get_snap_under_mouse(x, y, '+')
                print('placing to: %s, %s'%(dst_instance_id, dst_snap_id))
                if dst_instance_id and state['pick_snap'] is not None:
                    src_instance_id, src_snap_id = state['pick_snap']
                    src_instance = scene.instances[src_instance_id]
                    src_transform = src_instance.get_snap(
                            src_snap_id).transform
                    dst_transform = scene.instances[dst_instance_id].get_snap(
                            dst_snap_id).transform
                    if numpy.linalg.det(dst_transform[:3,:3]) < 0.:
                        dst_transform[0:3,0] *= -1
                    
                    best_transform = None
                    best_distance = -float('inf')
                    for i in range(4):
                        theta = i * math.pi/2
                        c = math.cos(theta)
                        s = math.sin(theta)
                        r = numpy.array([
                                [ c, 0, s, 0],
                                [ 0, 1, 0, 0],
                                [-s, 0, c, 0],
                                [ 0, 0, 0, 1]])
                        new_transform = (
                                dst_transform @
                                r @
                                numpy.linalg.inv(src_transform) @
                                src_instance.transform)
                        offset = (
                                new_transform @
                                numpy.linalg.inv(src_instance.transform))
                        pseudo_angle = numpy.trace(offset[:3,:3])
                        if pseudo_angle > best_distance:
                            best_transform = new_transform
                            best_distance = pseudo_angle
                    
                    scene.move_instance(
                            src_instance, best_transform)
                    
                    '''
                    offset = dst_transform @ numpy.linalg.inv(src_transform)
                    scene.move_instance(
                            src_instance, offset @ src_instance.transform)
                    '''
                    
            
            state['pick_snap'] = None
    
    def get_snap_under_mouse(x, y, polarity=None):
        
        # get matching snaps
        snaps = scene.get_snaps(polarity=polarity)
        snap_instance_names = [
                '%s_%i'%(inst, snap) for inst, snap in snaps]
        
        # prep scene
        bg_color = renderer.get_background_color()
        renderer.set_background_color((0,0,0))
        hidden = [renderer.instance_hidden(snap_name)
                for snap_name in snap_instance_names]
        for snap_name in snap_instance_names:
            renderer.show_instance(snap_name)
        
        # first render: instance id
        instance_id_lookup = {
                name : int(inst)
                for name, (inst, snap) in zip(snap_instance_names, snaps)}
        renderer.set_instance_masks_to_instance_indices(
                instance_id_lookup)
        part_mask_frame.enable()
        renderer.mask_render(instances=snap_instance_names)
        instance_map = part_mask_frame.read_pixels()
        instance_id = masks.color_byte_to_index(instance_map[y, x])
        
        # second render: snap id
        snap_id_lookup = {
                name : int(snap)
                for name, (inst, snap) in zip(snap_instance_names, snaps)}
        renderer.set_instance_masks_to_instance_indices(snap_id_lookup)
        part_mask_frame.enable()
        renderer.mask_render(instances=snap_instance_names)
        snap_map = part_mask_frame.read_pixels()
        snap_id = masks.color_byte_to_index(snap_map[y, x])
        
        renderer.set_background_color(bg_color)
        for snap, snap_hidden in zip(snap_instance_names, hidden):
            if snap_hidden:
                renderer.hide_instance(snap)
        
        return (instance_id, snap_id)
    
    def transform_about_snap(instance_id, snap_id, transform):
        instance = scene.instances[instance_id]
        snap_transform = instance.get_snap(snap_id).transform
        prototype_transform = instance.brick_type.snaps[snap_id].transform
        instance_transform = (
                snap_transform @
                transform @
                numpy.linalg.inv(prototype_transform))
        scene.move_instance(instance, instance_transform)
    
    window.register_callbacks(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutKeyboardFunc = key_press,
            glutKeyboardUpFunc = key_release,
            glutSpecialFunc = special_key,
            glutMouseFunc = camera_control.mouse_button,
            glutMotionFunc = camera_control.mouse_move)
    
    drpy_glut.start_main_loop()
    
    '''
    drpy_glut.glut_state['initializer'].start_main_loop(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutKeyboardFunc = keypress,
            glutMouseFunc = camera_control.mouse_button,
            glutMotionFunc = camera_control.mouse_move)
    '''
    
    '''
    window.start_main_loop(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutKeyboardFunc = keypress,
            glutMouseFunc = camera_control.mouse_button,
            glutMotionFunc = camera_control.mouse_move)
    '''
