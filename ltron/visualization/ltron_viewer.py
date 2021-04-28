import time
import sys
import os

import numpy

import PIL.Image as Image

#import renderpy.buffer_manager_glut as buffer_manager
import renderpy.glut as drpy_glut
from renderpy.frame_buffer import FrameBufferWrapper
import renderpy.core as core
import renderpy.camera as camera
from renderpy.interactive_camera import InteractiveCamera
import renderpy.masks as masks
import renderpy.assets as drpy_assets

import ltron.config as config
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
        white_background = True,
        print_fps = False):
    
    '''
    if ':' in file_path:
        file_path, subdocument = file_path.split(':')
    else:
        subdocument = None
    
    if not os.path.exists(file_path):
        file_path = ldraw_paths.LDRAW_FILES[file_path]
    '''
    
    resolved_file_path, subdocument = resolve_subdocument(file_path)
    
    config_paths = '%s:%s'%(
                    config.paths['renderpy_assets_cfg'],
                    drpy_assets.default_assets_path)
    
    drpy_glut.initialize_glut()
    window = drpy_glut.GlutWindowWrapper(
            'Brick Viewer', width, height)
    
    scene = BrickScene(renderable=True, track_snaps=True)
    scene.load_image_light(
            image_light,
            diffuse_texture=image_light + '_dif',
            reflect_texture=image_light + '_ref',
            set_active=True)
    
    scene.renderer.scene_description['image_lights'][image_light][
            'offset_matrix'] = numpy.array([
            [0.8660254037844387, 0.49999999999999994, 0, 0],
            [-0.49999999999999994, 0.8660254037844387, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if white_background:
        scene.renderer.scene_description['image_lights'][image_light]['render_background'] = False
        scene.renderer.set_background_color((1,1,1))
        scene.renderer.set_ambient_color((0.1, 0.1, 0.1))
    
    window.set_active()
    window.enable_window()
    
    part_mask_frame = FrameBufferWrapper(width, height, anti_alias=False)
    
    camera_control = InteractiveCamera(window, scene.renderer)
    
    state = {
        'render_mode' : 'color',
        'steps' : 0,
        'recent_file_change_time' : -1,
        'part_mask' : None,
        'snap_mode' : 'none',
        'batch_time' : time.time()
    }
    
    def reload_scene():
        while True:
            try:
                change_time = os.stat(resolved_file_path).st_mtime
                if change_time != state['recent_file_change_time']:
                    camera_pose = scene.get_camera_pose()
                    scene.instances.clear()
                    scene.import_ldraw(file_path)
                    
                    #renderer.load_scene(scene, clear_scene=True)
                    if state['recent_file_change_time'] == -1:
                        scene.camera_frame_scene(azimuth=-0.6, elevation=-0.3)
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
        
        part_mask_frame.enable()
        scene.mask_render(flip_y=True)
        state['part_mask'] = part_mask_frame.read_pixels()
        
        window.enable_window()
        if state['render_mode'] == 'color':
            scene.color_render(flip_y=False)
        if state['render_mode'] == 'removable':
            scene.removable_render(flip_y=False)
        elif state['render_mode'] == 'mask':
            scene.mask_render(flip_y=False)
    
    def get_instance_at_location(x, y):
        color = tuple(state['part_mask'][y,x])
        if color == (0,0,0):
            return None
        instance_id = masks.color_byte_to_index(color)
        return instance_id
    
    def keypress(key, x, y):
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
                instance = scene.get_instance(instance_id)
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
                scene.hide_instance(instance_id)
                print('Hiding Brick Instance %i'%instance_id)
        
        elif key == b'H':
            for instance in scene.instances:
                scene.hide_instance(instance)
        
        elif key == b'v':
            scene.show_all_instances()
            print('----')
            print('Showing All Hidden Instances')
        
        elif key == b'w':
            pixels = window.read_pixels()
            image_path = './brick_viewer_%06i.png'%state['steps']
            print('----')
            print('Writing Image To: %s'%image_path)
            Image.fromarray(numpy.flip(pixels, axis=0)).save(image_path)
        
        elif key == b'd':
            depth_map = part_mask_frame.read_pixels(
                    read_depth=True, projection=scene.renderer.get_projection())
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
                scene.show_all_snaps()
            else:
                state['snap_mode'] = 'none'
                scene.hide_all_snaps()
        
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
    
    window.start_main_loop(
            glutDisplayFunc = render,
            glutIdleFunc = render,
            glutKeyboardFunc = keypress,
            glutMouseFunc = camera_control.mouse_button,
            glutMotionFunc = camera_control.mouse_move)
    
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