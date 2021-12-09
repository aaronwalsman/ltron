import math
import os
import glob
import json

import numpy

from scipy.optimize import linear_sum_assignment

from pyquaternion import Quaternion

from PIL import Image

import tqdm
tqdm.monitor_interval = 0

from splendor.camera import orthographic_matrix
from splendor.frame_buffer import FrameBufferWrapper
from splendor.exceptions import (
    SplendorAssetException,
    SplendorEmptyMeshException,
)

import ltron.settings as settings
from ltron.home import get_ltron_home
from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.brick_type import BrickType
from ltron.ldraw.parts import LDRAW_PARTS, LDRAW_BLACKLIST_ALL
from ltron.geometry.utils import metric_close_enough, vector_angle_close_enough

symmetry_table_path = os.path.join(
    get_ltron_home(), 'symmetry_table.json')
LDRAW_SYMMETRY = {}
if os.path.exists(symmetry_table_path):
    with open(symmetry_table_path, 'r') as f:
        LDRAW_SYMMETRY.update(json.load(f))

default_resolution = 512
default_tolerance = 1

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
        projection = orthographic_matrix(l=l, r=r, b=b, t=t, n=n, f=f)
        scene.set_projection(projection)
        scene.set_view_matrix(numpy.linalg.inv(camera_pose))
        
        frame_buffer.enable()
        scene.viewport_scissor(0,0,frame_buffer.width, frame_buffer.height)
        scene.mask_render(instances=[str(instance)])
        a_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=projection)
        
        scene.move_instance(instance, pose_b)
        scene.viewport_scissor(0,0,frame_buffer.width, frame_buffer.height)
        scene.mask_render(instances=[str(instance)])
        b_depth_map = frame_buffer.read_pixels(
            read_depth=True, projection=projection)
        
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
        
        if not numpy.all(close):
            cleanup_scene()
            return False
    
    cleanup_scene()
    return True

symmetry_offsets = {
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
    
    # if the brick has no shape, return nothing
    if brick_type.empty_shape:
        return []
    
    try:
        # Hack scale because I think the blender ImportLdraw plugin is
        # ever-so-slightly wrong with how it handles gaps
        # TODO: I should probably go in and fix all the obj files to compensate
        # for this instead of this hack.
        bbox_min, bbox_max = brick_type.bbox
        bbox_range = bbox_max - bbox_min
        
        xyz_scale = []
        for i in range(3):
            if bbox_range[i] == 0.:
                xyz_scale.append(1.)
            else:
                s = (bbox_range[0] + 0.5) / bbox_range[0]
                if s > 1. / 0.95:
                    s = 1. / 0.95
                xyz_scale.append(s)
        
        '''
        if bbox_range[0] == 0.:
            x_scale = 1.
        else:
            x_scale = (bbox_range[0] + 0.5) / bbox_range[0]
            if x_scale > 1. / 0.95:
                x_scale = 1. / 0.95
        if bbox_range[1] == 0.:
            y_scale = 1.
        else:
            y_scale = (bbox_range[1] + 0.5) / bbox_range[1]
            if y_scale > 1. / 0.95:
                y_scale = 1. / 0.95
        if bbox_range[2] == 0.
            z_scale = 1.
        else:
            z_scale = (bbox_range[2] + (0.5*0.33)) / bbox_range[2]
            if z_scale > 1. / 0.95:
                z_scale = 1. / 0.95
        '''
        
        scale = numpy.eye(4)
        scale[0,0] = xyz_scale[0] #x_scale
        scale[1,1] = xyz_scale[1] #y_scale
        scale[2,2] = xyz_scale[2] #z_scale
        default_pose = scale
        
        centroid = numpy.mean(brick_type.bbox_vertices, axis=1)
        translate = numpy.eye(4)
        translate[:3,3] = -centroid[:3]
        default_pose = translate @ default_pose
        
        symmetries = []
        
        try:
            for name, symmetry_offset in symmetry_offsets.items():
                test_pose = symmetry_offset @ default_pose
                if check_single_symmetry(
                    brick_type,
                    scene,
                    framebuffer,
                    default_pose,
                    test_pose,
                    tolerance,
                    label = name,
                ):
                    symmetries.append(name)
        except SplendorEmptyMeshException:
            pass
        
        return symmetries
    
    except:
        raise
        return 'FAIL (EXCEPTION)'

def build_symmetry_table(
    bricks=None,
    symmetry_table_path=symmetry_table_path,
    resolution=default_resolution,
    tolerance=default_tolerance,
    error_handling='raise',
):
    #all_brick_types = glob.glob(
    #    os.path.join(settings.paths['ldraw'], 'parts', '*.dat'))
    
    if bricks is None:
        bricks = LDRAW_PARTS
    bricks = set(bricks) - LDRAW_BLACKLIST_ALL
    
    symmetry_table = {}
    scene = BrickScene(renderable=True)
    framebuffer = FrameBufferWrapper(
        resolution, resolution, anti_alias=False)
    iterate = tqdm.tqdm(bricks)
    for brick_type in iterate:
        brick_name = os.path.split(brick_type)[-1]
        iterate.set_description(brick_name.ljust(20))
        brick_type = os.path.split(brick_type)[-1]
        if error_handling == 'skip':
            try:
                symmetry_table[brick_type] = check_bricktype_symmetry(
                    brick_type, scene, framebuffer, tolerance)
            except SplendorAssetException:
                print('Could not find brick "%s"'%brick_type)
            except KeyboardInterrupt:
                break
            except:
                print('Error for brick "%s"'%brick_type)
        elif error_handling == 'raise':
            try:
                symmetry_table[brick_type] = check_bricktype_symmetry(
                    brick_type, scene, framebuffer, tolerance)
            except:
                print('Error for brick "%s"'%brick_type)
                raise
        else:
            raise ValueError('"error_handling" must be "skip" or "raise"')
        
        scene.clear_instances()
    
    symmetry_table_path = os.path.join(get_ltron_home(), 'symmetry_table.json')
    with open(symmetry_table_path, 'w') as f:
        json.dump(symmetry_table, f, indent=2)

symmetry_tests = {
    'rx90':([1,0,0], math.radians(90.)),
    'rx180':([1,0,0], math.radians(180.)),
    'ry90':([0,1,0], math.radians(90.)),
    'ry180':([0,1,0], math.radians(180.)),
    'rz90':([0,0,1], math.radians(90.)),
    'rz180':([0,0,1], math.radians(180.)),
}

def pose_match_under_symmetries(
    symmetries,
    pose_a,
    pose_b,
    metric_tolerance=1.,
    angular_tolerance=0.08,
):
    if not metric_close_enough(pose_a[:3,3], pose_b[:3,3], metric_tolerance):
        return False
    
    r_ab = pose_a[:3,:3].T @ pose_b[:3,:3]
    q = Quaternion(matrix=r_ab)
    if abs(q.angle) < angular_tolerance:
        return True
    
    for symmetry in symmetries:
        symmetry_axis, symmetry_angle = symmetry_tests[symmetry]
        if vector_angle_close_enough(
            symmetry_axis, q.axis, angular_tolerance, allow_negative=True
        ):
            angle_offset = abs(
                round(q.angle / symmetry_angle) * symmetry_angle - q.angle)
            if angle_offset < angular_tolerance:
                return True
    
    return False

def brick_pose_match_under_symmetry(
    part_name,
    pose_a,
    pose_b,
    metric_tolerance=1.,
    angular_tolerance=0.08,
):
    symmetries = LDRAW_SYMMETRY[part_name]
    return pose_match_under_symmetries(
        symmetries, pose_a, pose_b, metric_tolerance, angular_tolerance)
