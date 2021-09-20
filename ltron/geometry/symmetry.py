import time
import os
import glob
import json

import numpy

from scipy.optimize import linear_sum_assignment

from PIL import Image

import tqdm
tqdm.monitor_interval = 0

from splendor.camera import orthographic_matrix
from splendor.frame_buffer import FrameBufferWrapper
from splendor.exceptions import SplendorAssetException

import ltron.settings as settings
from ltron.home import get_ltron_home
from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.brick_type import BrickType
from ltron.geometry.grid_bucket import GridBucket

default_tolerance = 1

'''
def check_single_symmetry(
    vertices,
    pose_a,
    pose_b,
    tolerance=default_tolerance,
    parallel_vertices=16,
    label='',
):
    vertices_a = pose_a @ vertices
    vertices_b = pose_b @ vertices
    
    #vertices_a = vertices_a.reshape(4, 1, -1)
    #vertices_b = vertices_b.reshape(4, -1, 1)
    #squared_distance = numpy.sum((vertices_a - vertices_b)[:3]**2, axis=0)
    #min_distances = numpy.min(squared_distance, axis=0)
    #return numpy.all(min_distances <= tolerance**2)
    
    #bucket = GridBucket(tolerance/4)
    #bucket.insert_many(range(vertices_a.shape[1]), vertices_a[:3].T)
    #nearest = bucket.lookup_many(vertices_b[:3].T, tolerance)
    #return all(nearest)
    
    vertices_b = vertices_b.reshape(4, -1, 1)
    iterate = tqdm.tqdm(range(0, vertices_a.shape[1], parallel_vertices))
    iterate.set_description(label)
    for i in iterate:
        chunk = vertices_a[:,i:i+parallel_vertices].reshape(4,1,-1)
        offsets = chunk - vertices_b
        squared_distances = numpy.sum(offsets**2, axis=0)
        min_squared_distances = numpy.min(squared_distances, axis=0)
        if not numpy.all(min_squared_distances <= tolerance**2):
            return False
    
    return True
'''

def check_single_symmetry(
    brick_type,
    scene,
    frame_buffer,
    pose_a,
    pose_b,
    tolerance=default_tolerance,
    label='',
):
    original_view_matrix = scene.get_view_matrix()
    original_projection = scene.get_projection()
    
    # add new instance
    instance = scene.add_instance(brick_type, 1, pose_a)
    
    def cleanup_scene():
        scene.remove_instance(instance)
        scene.set_view_matrix(original_view_matrix)
        scene.set_projection(original_projection)
    
    for i, camera_pose in enumerate((
        numpy.array([ # 0
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]]),
        numpy.array([ # 90 y
            [ 0, 0, 1, 0],
            [ 0, 1, 0, 0],
            [-1, 0, 0, 0],
            [ 0, 0, 0, 1]]),
        numpy.array([ # 180 y
            [-1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0,-1, 0],
            [ 0, 0, 0, 1]]),
        numpy.array([ # 270 y
            [ 0, 0,-1, 0],
            [ 0, 1, 0, 0],
            [ 1, 0, 0, 0],
            [ 0, 0, 0, 1]]),
        numpy.array([ # 90 x
            [ 1, 0, 0, 0],
            [ 0, 0,-1, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 0, 1]]),
        numpy.array([ # 270 x
            [ 1, 0, 0, 0],
            [ 0, 0, 1, 0],
            [ 0,-1, 0, 0],
            [ 0, 0, 0, 1]]),
    )):
        scene.move_instance(instance, pose_a)
        local_a_vertices = (
            numpy.linalg.inv(camera_pose) @ instance.bbox_vertices())
        v_min = numpy.min(local_a_vertices, axis=1)
        v_max = numpy.max(local_a_vertices, axis=1)
        translate = numpy.eye(4)
        translate[2,3] = v_max[2] + 2
        camera_pose = camera_pose @ translate
        near_clip = 1
        far_clip = v_max[2] + 3
        l = v_max[0] + 20
        r = v_min[0] - 20
        b = -v_max[1] - 20
        t = -v_min[1] + 20
        n = near_clip
        f = far_clip
        #print(label)
        #print(l, r, b, t, n, f)
        #print(v_min, v_max)
        projection = orthographic_matrix(l=l, r=r, b=b, t=t, n=n, f=f)
        #print('proj')
        #print(projection)
        #print('pose')
        #print(camera_pose)
        #print('a/b')
        #print(pose_a)
        #print(pose_b)
        scene.set_projection(projection)
        scene.set_view_matrix(numpy.linalg.inv(camera_pose))
        
        frame_buffer.enable()
        scene.viewport_scissor(0,0,frame_buffer.width, frame_buffer.height)
        scene.mask_render(instances=[str(instance)])
        a_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=projection)
        #a_color = frame_buffer.read_pixels()
        #Image.fromarray(a_color[1:-1,1:-1]).save('test_%s_%i_0.png'%(label, i))
        
        scene.move_instance(instance, pose_b)
        scene.viewport_scissor(0,0,frame_buffer.width, frame_buffer.height)
        scene.mask_render(instances=[str(instance)])
        b_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=projection)
        #b_color = frame_buffer.read_pixels()
        #Image.fromarray(b_color[1:-1,1:-1]).save('test_%s_%i_1.png'%(label, i))
        
        b_stack = numpy.concatenate((
            b_depth_map[0:-2,0:-2],
            b_depth_map[0:-2,1:-1],
            b_depth_map[0:-2,2:  ],
            b_depth_map[1:-1,0:-2],
            b_depth_map[1:-1,1:-1],
            b_depth_map[1:-1,2:  ],
            b_depth_map[2:  ,0:-2],
            b_depth_map[2:  ,1:-1],
            b_depth_map[2:  ,2:  ],
        ), axis=-1)
        
        offsets = numpy.abs(a_depth_map[1:-1,1:-1] - b_stack)
        close = numpy.any(offsets <= tolerance, axis=-1)
        
        #import pdb
        #pdb.set_trace()
        
        if not numpy.all(close):
            cleanup_scene()
            return False
    
    cleanup_scene()
    return True


symmetry_tests = {
    'rx90':numpy.array([
        [ 1, 0, 0, 0],
        [ 0, 0,-1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]]),
    'rx180':numpy.array([
        [ 1, 0, 0, 0],
        [ 0,-1, 0, 0],
        [ 0, 0,-1, 0],
        [ 0, 0, 0, 1]]),
    'ry90':numpy.array([
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [-1, 0, 0, 0],
        [ 0, 0, 0, 1]]),
    'ry180':numpy.array([
        [-1, 0, 0, 0],
        [ 0, 1, 0, 0],
        [ 0, 0,-1, 0],
        [ 0, 0, 0, 1]]),
    'rz90':numpy.array([
        [ 0,-1, 0, 0],
        [ 1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1]]),
    'rz180':numpy.array([
        [-1, 0, 0, 0],
        [ 0,-1, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1]]),
}

def check_bricktype_symmetry(
    brick_type,
    scene,
    framebuffer,
    tolerance=default_tolerance
):
    if isinstance(brick_type, str):
        brick_type = BrickType(brick_type)
    
    #if not brick_type.vertices.shape[1]:
    #    return 'FAIL (NO VERTICES)'
    
    try:
        #centroid = numpy.mean(brick_type.bbox_vertices, axis=1)
        
        #default_pose = numpy.eye(4)
        #default_pose[:3,3] = -centroid[:3]
        
        # Hack scale because I think the blender ImportLdraw plugin is
        # ever-so-slightly wrong with how it handles gaps
        # TODO: I should probably go in and fix all the obj files to compensate
        # for this instead of this hack.
        bbox_min, bbox_max = brick_type.bbox
        bbox_range = bbox_max - bbox_min
        x_scale = (bbox_range[0] + 0.5) / bbox_range[0]
        if x_scale > 1. / 0.95:
            x_scale = 1. / 0.95
        y_scale = (bbox_range[1] + 0.5) / bbox_range[1]
        if y_scale > 1. / 0.95:
            y_scale = 1. / 0.95
        z_scale = (bbox_range[2] + (0.5*0.33)) / bbox_range[2]
        if z_scale > 1. / 0.95:
            z_scale = 1. / 0.95
        scale = numpy.eye(4)
        scale[0,0] = x_scale
        scale[1,1] = y_scale
        scale[2,2] = z_scale
        #default_pose = default_pose @ scale
        default_pose = scale
        
        #scaled_bbox = scale @ brick_type.bbox_vertices
        centroid = numpy.mean(brick_type.bbox_vertices, axis=1)
        translate = numpy.eye(4)
        translate[:3,3] = -centroid[:3]
        default_pose = translate @ default_pose
        
        symmetries = []
        for name, symmetry_test in symmetry_tests.items():
            test_pose = symmetry_test @ default_pose
            if check_single_symmetry(
                #brick_type.vertices,
                brick_type,
                scene,
                framebuffer,
                default_pose,
                test_pose,
                tolerance,
                label = name,
                #label='%s [%s]'%(str(brick_type), name),
            ):
                symmetries.append(name)
        
        return symmetries
    
    except:
        raise
        return 'FAIL (EXCEPTION)'
        

def build_symmetry_table(resolution=512, tolerance=default_tolerance):
    all_brick_types = glob.glob(
        os.path.join(settings.paths['ldraw'], 'parts', '*.dat'))
    symmetry_table = {}
    scene = BrickScene(renderable=True)
    framebuffer = FrameBufferWrapper(
        resolution, resolution, anti_alias=False)
    iterate = tqdm.tqdm(all_brick_types)
    for brick_type in iterate:
        brick_name = os.path.split(brick_type)[-1]
        iterate.set_description(brick_name.ljust(20))
        '''
        brick_type = BrickType(part)
        if not brick_type.vertices.shape[1]:
            symmetry_table[part] = 'FAIL (NO VERTICES)'
        else:
            try:
                symmetry_table[part] = check_bricktype_symmetry(
                    brick_type, tolerance)
            except:
                symmetry_table[part] = 'FAIL (EXCEPTION)'
        '''
        brick_type = os.path.split(brick_type)[-1]
        try:
            symmetry_table[brick_type] = check_bricktype_symmetry(
                brick_type, scene, framebuffer, tolerance)
        except SplendorAssetException:
            print('Could not find brick "%s"'%brick_type)
        except KeyboardInterrupt:
            break
        except:
            print('Error for brick "%s"'%brick_type)
        
        scene.clear_instances()
    
    symmetry_table_path = os.path.join(get_ltron_home(), 'symmetry_table.json')
    with open(symmetry_table_path, 'w') as f:
        json.dump(symmetry_table, f, indent=2)

if __name__ == '__main__':
    build_symmetry_table()
    #scene = BrickScene(renderable=True)
    #framebuffer = FrameBufferWrapper(
    #    512,512, anti_alias=False)
    #print(check_bricktype_symmetry("32051.dat", scene, framebuffer))
