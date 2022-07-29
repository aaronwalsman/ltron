import time
import math
import sys
import os

import numpy

import PIL.Image as Image

from pyquaternion import Quaternion

import splendor.contexts.glut as glut
from splendor.frame_buffer import FrameBufferWrapper
import splendor.core as core
import splendor.camera as camera
from splendor.interactive_camera import InteractiveCamera
import splendor.masks as masks
from splendor.image import save_image

import ltron.settings as settings
from ltron.dataset.paths import resolve_subdocument
from ltron.bricks.brick_scene import BrickScene

instructions = '''
LTRON Viewer Hotkeys

reload -------------------------------------------------------------------------
l:                      reload scene

camera -------------------------------------------------------------------------
Left-Click and drag :   orbit camera
Right-Click and drag :  pan camera
scroll :                dolly camera in-and-out

rendering ----------------------------------------------------------------------
l :                     color render
m :                     mask render
s :                     snap render instance id
S :                     render snap id
- :                     set snap rendering to -
+ :                     set snap rendering to +

info ---------------------------------------------------------------------------
i :                     print brick instance id, part name and transform
                        of the brick under the cursor as well as snap id
                        and transform if rendering snaps
c :                     print connections to brick under the cursor

visibility ---------------------------------------------------------------------
h :                     hide the brick under the cursor
H :                     hide all bricks
v :                     show all bricks

output -------------------------------------------------------------------------
w :                     write out a color image to the current directory
d :                     write out a depth image to the current directory
e :                     export current scene
E :                     export the current scene by OVERWRITING THE LOADED SCENE

manipulation -------------------------------------------------------------------
hold p :                hover over a positive snap, then hold p,
                        then hover over a negative snap and release
                        to snap the positive snap to the negative one
hold n :                hover over a negative snap, then hold n,
                        then hover over a positive snap and release
                        to snap the negative snap to the positive one
. :                     rotate the brick belonging to the positive snap
                        under the cursor by 90 degrees
, :                     rotate the brick belonging to the positive snap
                        under the cursor by -90 degrees
> :                     rotate the brick belonging to the negative snap
                        under the cursor by 90 degrees
< :                     rotate the brick belonging to the negative snap
                        under the cursor by -90 degrees
[ :                     move the brick under the cursor one step in the
                        local -y direction
] :                     move the brick under the cursor one step in the
                        local y direction
left-arrow :            move the brick under the cursor one step in the
                        local -x direction
right-arrow :           move the brick under the cursor one step in the
                        local x direction
down-arrow :            move the brick under the cursor one step in the
                        local -z direction
up-arrow :              move the brick under the cursor one step in the
                        local z direction
'''

def start_viewer(
        file_path,
        width = 512,
        height = 512,
        image_light = 'grey_cube',
        background_color = (102, 102, 102),
        poll_frequency = 8,
        print_fps = False):
    
    resolved_file_path, subdocument = resolve_subdocument(file_path)
    
    scene = BrickScene(
        renderable=True,
        collision_checker=True,
        render_args={
            'opengl_mode':'glut',
            'window_width':width,
            'window_height':height,
            'load_scene':'front_light'},
        track_snaps=True)
    
    window = scene.render_environment.window
    renderer = scene.render_environment.renderer
    
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
    '''
    #scene.set_background_color((0.65,0.65,0.65))
    #scene.set_background_color((1.0, 1.0, 1.0))
    scene.set_background_color(tuple([b/255 for b in background_color]))
    
    window.set_active()
    window.enable_window()
    
    part_mask_frame = FrameBufferWrapper(width, height, anti_alias=False)
    
    camera_control = InteractiveCamera(window, renderer)
    
    state = {
        'render_mode' : 'color',
        'snap_id_mode' : 'instance',
        'snap_polarity' : '+',
        'steps' : 0,
        'recent_file_change_time' : -1,
        'part_mask' : None,
        'snap_mode' : 'none',
        'batch_time' : time.time(),
        'pick_snap' : None,
    }
    
    print(instructions)
    
    def reload_scene(force=False):
        while True:
            try:
                change_time = os.stat(resolved_file_path).st_mtime
                if change_time != state['recent_file_change_time'] or force:
                    t_start_load = time.time()
                    view_matrix = scene.get_view_matrix()
                    scene.instances.clear()
                    scene.import_ldraw(file_path)
                    
                    #renderer.load_scene(scene, clear_scene=True)
                    if state['recent_file_change_time'] == -1:
                        scene.camera_frame_scene(azimuth=-2.5, elevation=-0.3)
                    else:
                        scene.set_view_matrix(view_matrix)
                    state['recent_file_change_time'] = change_time
                    t_end_load = time.time()
                    print('Loaded: %s'%file_path)
                    print('Elapsed: %f'%(t_end_load-t_start_load))
                    print('Shapes: %i'%len(scene.shape_library))
                    print('Colors: %i'%len(scene.color_library))
                    print('Brick Instances: %i'%len(scene.instances))
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
        
        brick_instances = list(scene.instances.keys())
        brick_instances = [str(b) for b in brick_instances]
        
        window.enable_window()
        if state['render_mode'] == 'color':
            scene.color_render(flip_y=False, instances=brick_instances)
        elif state['render_mode'] == 'removable':
            scene.removable_render(flip_y=False)
        elif state['render_mode'] == 'mask':
            scene.mask_render(flip_y=False, instances=brick_instances)
        elif state['render_mode'] == 'snap':
            snaps = scene.get_matching_snaps(
                polarity=state['snap_polarity'])
            if state['snap_id_mode'] == 'instance':
                scene.snap_render_instance_id(snaps, flip_y=False)
            elif state['snap_id_mode'] == 'snap':
                scene.snap_render_snap_id(snaps, flip_y=False)
    
    def get_instance_at_location(x, y):
        brick_instances = list(scene.instances.keys())
        brick_instances = [str(b) for b in brick_instances]
        part_mask_frame.enable()
        background_color = scene.get_background_color()
        scene.set_background_color((0,0,0))
        scene.mask_render(flip_y=True, instances=brick_instances)
        state['part_mask'] = part_mask_frame.read_pixels()
        
        color = tuple(state['part_mask'][y,x])
        if color == (0,0,0):
            scene.set_background_color(background_color)
            return None
        instance_id = masks.color_byte_to_index(color)
        
        scene.set_background_color(background_color)
        
        return instance_id
    
    def key_press(key, x, y):
        if key == b'l':
            state['render_mode'] = 'color'
        elif key == b'm':
            #if state['render_mode'] != 'mask':
            #    state['render_mode'] = 'mask'
            #else:
            #    state['render_mode'] = 'color'
            state['render_mode'] = 'mask'
        
        #if key == b'r':
        #    if state['render_mode'] != 'removable':
        #        state['render_mode'] = 'removable'
        #    else:
        #        state['render_mode'] = 'color'
        
        elif key == b'i':
            if state['render_mode'] == 'snap':
                instance_id, snap_id = get_snap_under_mouse(
                    x, y, state['snap_polarity'])
            #elif state['render_mode'] == 'snap':
            #    instance_id, snap_id = get_snap_under_mouse(x, y, '-')
            else:
                instance_id = get_instance_at_location(x, y)
                snap_id = None
            
            print('----')
            if instance_id is None:
                print('No Part Selected')
            else:
                print('Instance ID: %i'%instance_id)
                instance = scene.instances[instance_id]
                transform = instance.transform
                type_name = instance.brick_shape.reference_name
                print('Part Name: %s'%type_name)
                #print('Translation: %f, %f, %f'%(
                #        transform[0,3],
                #        transform[1,3],
                #        transform[2,3]))
                print('Instance Transform:')
                print(transform)
                
                if snap_id is not None:
                    print('Snap ID: %i'%snap_id)
                    snap = instance.snaps[snap_id]
                    transform = snap.transform
                    print('Snap Transform:')
                    print(transform)
        
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
            second_max_depth, max_depth = numpy.unique(depth_map)[-2:]
            #depth_map = 1.0 - (depth_map - min_depth) / (500 - min_depth)
            depth_map = 1.0 - (depth_map - min_depth) / (second_max_depth - min_depth)
            depth_map = numpy.clip(depth_map, 0, 1)
            depth_map = (depth_map * 255).astype(numpy.uint8)
            print(depth_map.shape)
            
            image_path = './brick_viewer_depth_%06i.png'%state['steps']
            print('----')
            print('Writing Depth Image To: %s'%image_path)
            Image.fromarray(depth_map[...,0]).save(image_path)
        
        elif key == b's':
            state['render_mode'] = 'snap'
            state['snap_id_mode'] = 'instance'
            #if (state['render_mode'] == 'snap' and
            #    state['snap_index_mode'] == 'instance'
            #):
            #    state['render_mode'] = 'color'
            #else:
            #    state['render_mode'] = 'snap'
            #    state['snap_index_mode'] = 'instance'
            #if state['snap_mode'] == 'none':
                #state['snap_mode'] = 'all'
                #scene.show_all_snap_instances()
            #else:
                #state['snap_mode'] = 'none'
                #scene.hide_all_snap_instances()
        
        elif key == b'-' or key == b'_':
            state['snap_polarity'] = '-'
        
        elif key == b'=' or key == b'+':
            state['snap_polarity'] = '+'
        
        elif key == b'S':
            state['render_mode'] = 'snap'
            state['snap_id_mode'] = 'snap'
            #if (state['render_mode'] == 'snap' and
            #    state['snap_index_mode'] == 'snap'
            #):
            #    state['render_mode'] = 'color'
            #else:
            #    state['render_mode'] = 'snap+'
        
        elif key == b'c':
            instance_id = get_instance_at_location(x, y)
            print('----')
            if instance_id is None:
                print('No Part Selected')
                return
            
            connected_snaps = scene.get_instance_snap_connections(
                    str(instance_id))
            print('Instance: %i'%instance_id)
            print('All Connections:')
            for snap_a, snap_b in connected_snaps:
                print(snap_a, ':', snap_b)
            
            connected_instances = set(
                int(snap_b.brick_instance)
                for snap_a, snap_b in connected_snaps
            )
            
            print('All Connected Instances:')
            print(list(sorted(connected_instances)))
        
        elif key == b'p' or key == b'P' or key == b'n' or key == b'N':
            if key == b'p' or key == b'P':
                # render positive snaps in mask mode
                instance_id, snap_id = get_snap_under_mouse(x, y, '+')
                print('picked: %s, %s'%(instance_id, snap_id))
                if instance_id:
                    state['pick_snap'] = (instance_id, snap_id)
            
            if key == b'n' or key == b'N':
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
            if instance_id is None:
                print('No Part Selected')
                return
            transform = scene.instances[instance_id].transform
            #transform[1,3] -= 8
            transform = transform @ numpy.array([
                [1, 0, 0, 0],
                [0, 1, 0, -8],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
            scene.move_instance(instance_id, transform)
        
        elif key == b']':
            instance_id = get_instance_at_location(x, y)
            if instance_id is None:
                print('No Part Selected')
                return
            transform = scene.instances[instance_id].transform
            #transform[1,3] += 8
            transform = transform @ numpy.array([
                [1, 0, 0, 0],
                [0, 1, 0, 8],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
            scene.move_instance(instance_id, transform)
    
    def special_key(key, x, y):
        if key == glut.GLUT.GLUT_KEY_DOWN:
            instance_id = get_instance_at_location(x, y)
            if instance_id is None:
                print('No Part Selected')
                return
            transform = scene.instances[instance_id].transform
            #transform[2,3] -= 10
            transform = transform @ numpy.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -20],
                [0, 0, 0, 1]])
            scene.move_instance(instance_id, transform)
        elif key == glut.GLUT.GLUT_KEY_UP:
            instance_id = get_instance_at_location(x, y)
            if instance_id is None:
                print('No Part Selected')
                return
            transform = scene.instances[instance_id].transform
            #transform[2,3] += 10
            transform = transform @ numpy.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 20],
                [0, 0, 0, 1]])
            scene.move_instance(instance_id, transform)
        elif key == glut.GLUT.GLUT_KEY_LEFT:
            instance_id = get_instance_at_location(x, y)
            if instance_id is None:
                print('No Part Selected')
                return
            transform = scene.instances[instance_id].transform
            #transform[0,3] -= 10
            transform = transform @ numpy.array([
                [1, 0, 0, -20],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
            scene.move_instance(instance_id, transform)
        elif key == glut.GLUT.GLUT_KEY_RIGHT:
            instance_id = get_instance_at_location(x, y)
            if instance_id is None:
                print('No Part Selected')
                return
            transform = scene.instances[instance_id].transform
            #transform[0,3] += 10
            transform = transform @ numpy.array([
                [1, 0, 0, 20],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
            scene.move_instance(instance_id, transform)
    
    def key_release(key, x, y):
        if key == b'p' or key == b'n':
            # render negative snaps in mask mode
            if key == b'p':
                p = '-'
            else:
                p = '+'
            place_snap = get_snap_under_mouse(x, y, p)
            if place_snap[0] is not None and state['pick_snap'] is not None:
                print('placing to: %s, %s'%(place_snap[0], place_snap[1]))
                i, s = state['pick_snap']
                pick_snap = scene.instances[i].snaps[s]
                i, s = place_snap
                place_snap = scene.instances[i].snaps[s]
                scene.pick_and_place_snap(
                    pick_snap, place_snap, check_collision=True)
                state['pick_snap'] = None
            
            '''
            elif key == b'n':
                # render positive snaps in mask mode
                place_snap = get_snap_under_mouse(x, y, '+')
                if place_snap[0] is not None and state['pick_snap'] is not None:
                    print('placing to: %s, %s'%(place_snap[0], place_snap[1]))
                    i, s = state['pick_snap']
                    pick_snap = scene.instances[i].snaps[s]
                    i, s = place_snap
                    place_snap = scene.instances[i].snaps[s]
                    scene.pick_and_place_snap(
                        pick_snap, place_snap, check_collision=True)
                    state['pick_snap'] = None
            '''
            state['pick_snap'] = None
        
        
        elif key == b'P' or key == b'N':
            # render negative snaps in mask mode
            if key == b'p':
                p = '-'
            else:
                p = '+'
            place_snap = get_snap_under_mouse(x, y, p)
            if place_snap[0] is not None and state['pick_snap'] is not None:
                print('placing to: %s, %s'%(place_snap[0], place_snap[1]))
                i, s = state['pick_snap']
                pick_snap = scene.instances[i].snaps[s]
                j, s = place_snap
                place_snap = scene.instances[j].snaps[s]
                
                pick_and_place_transforms = (
                    scene.all_pick_and_place_transforms(
                        pick_snap, place_snap, check_collision=False
                    )
                )
                instance_transform = scene.instances[i].transform
                for transform in pick_and_place_transforms:
                    scene.move_instance(i, transform)
                    render()
                    time.sleep(1)
                
                scene.move_instance(i, instance_transform)
                
                state['pick_snap'] = None
    
    def get_snap_under_mouse(x, y, polarity=None):
        
        # get matching snaps
        snaps = scene.get_matching_snaps(polarity=polarity)
        #snap_instance_names = [
        #        '%s_%i'%(inst, snap) for inst, snap in snaps]
        snap_instance_names = [str(snap) for snap in snaps]
        
        # prep scene
        bg_color = renderer.get_background_color()
        renderer.set_background_color((0,0,0))
        hidden = [renderer.instance_hidden(str(snap)) for snap in snaps]
        for snap in snaps:
            renderer.show_instance(str(snap))
        
        # first render: instance id
        #instance_id_lookup = {
        #        name : int(inst)
        #        for name, (inst, snap) in zip(snap_instance_names, snaps)}
        instance_id_lookup = {
            str(snap) : int(snap.brick_instance) for snap in snaps}
        renderer.set_instance_masks_to_instance_indices(
                instance_id_lookup)
        part_mask_frame.enable()
        renderer.mask_render(instances=snap_instance_names)
        instance_map = part_mask_frame.read_pixels()
        instance_id = masks.color_byte_to_index(instance_map[y, x])
        
        # second render: snap id
        #snap_id_lookup = {
        #        name : int(snap)
        #        for name, (inst, snap) in zip(snap_instance_names, snaps)}
        snap_id_lookup = {
            str(snap) : int(snap.snap_style) for snap in snaps}
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
        snap_transform = instance.snaps[snap_id].transform
        prototype_transform = instance.brick_shape.snaps[snap_id].transform
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
    
    glut.start_main_loop()
    
    '''
    glut.glut_state['initializer'].start_main_loop(
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
