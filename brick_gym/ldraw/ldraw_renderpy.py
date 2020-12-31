import math
import os

import numpy

import renderpy.camera as camera
import renderpy.masks as masks

import brick_gym.ldraw.mpd as mpd
import brick_gym.ldraw.colors as colors

import brick_gym.config as config

upright = numpy.array([
    [-1, 0, 0, 0],
    [ 0,-1, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, 0, 0, 1]])

def document_to_renderpy(
        document,
        image_light = None,
        ambient_color = (0,0,0)):
    
    # setup renderpy data
    renderpy_data = {
        'image_lights' : {},
        'active_image_light' : 'background',
        'meshes': {},
        'materials' : {},
        'instances' : {},
        'ambient_color' : ambient_color,
        'point_lights' : {},
        'direction_lights' : {},
    }
    
    # add meshes
    obj_directory = config.paths['obj']
    
    parts = document.get_all_parts()
    unique_bricks = set(part[0] for part in parts)
    unique_objs = [brick + '.obj' for brick in unique_bricks]
    
    # add image light
    if image_light_directory is not None:
        renderpy_data['image_lights']['background'] = {
            'texture_directory' : image_light,
            'reflection_mipmaps' : None,
            'blur' : 0,
            'render_background' : 1,
            'diffuse_contrast' : 1,
            'rescale_diffuse_intensity' : False
        }

def mpd_to_renderpy(mpd_data,
        image_light_directory = None,
        ambient_color = (0,0,0)):
    
    obj_directory = config.paths['obj']
    external_parts = os.listdir(os.path.join(config.paths['ldraw'], 'parts'))
    
    parts, complete = mpd.parts_from_mpd(mpd_data, external_parts)
    unique_dats = set(part['file_name'] for part in parts)
    unique_objs = [
            os.path.splitext(unique_dat)[0] + '.obj'
            for unique_dat in unique_dats]
    
    renderpy_data = {
        'image_lights' : {},
        'active_image_light' : 'background',
        'meshes': {},
        'materials' : {},
        'instances' : {},
        'ambient_color' : ambient_color,
        'point_lights' : {},
        'direction_lights' : {},
    }
    
    if image_light_directory is not None:
        renderpy_data['image_lights']['background'] = {
            'texture_directory' : image_light_directory,
            'reflection_mipmaps' : None,
            'blur' : 0,
            'render_background' : 1,
            'diffuse_contrast' : 1,
            'rescale_diffuse_intensity' : False
        }
    
    # meshes
    part_count = {}
    for unique_obj in unique_objs:
        mesh_name = os.path.splitext(unique_obj)[0]
        part_count[mesh_name] = 0
        #mesh_path = os.path.abspath(os.path.join(obj_directory, unique_obj))
        renderpy_data['meshes'][mesh_name] = {
            #'mesh_path' : mesh_path,
            'mesh_asset' : mesh_name,
            'scale' : 1.0,
            'create_uvs' : True
        }
    
    # materials
    unique_colors = set(part['color'] for part in parts)
    for unique_color in unique_colors:
        unique_color = int(unique_color)
        if unique_color in colors.color_index_to_alt_rgb:
            color = colors.color_index_to_alt_rgb[unique_color]
        else:
            color = (128, 128, 128)
        texture = numpy.ones((16,16,3))
        texture[:,:,0] = color[0]
        texture[:,:,1] = color[1]
        texture[:,:,2] = color[2]
        texture = texture.tolist()
        color_name = 'mat_%i'%unique_color
        renderpy_data['materials'][color_name] = {
            'texture' : texture,
            'ka' : 1.0,
            'kd' : 0.0,
            'ks' : 0.0,
            'shine' : 1.0,
            'image_light_kd' : 0.75,
            'image_light_ks' : 0.25,
            'image_light_blur_reflection' : 2.0
        }
    
    # instances
    min_x = float('inf')
    min_y = float('inf')
    min_z = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    max_z = float('-inf')
    for i, part in enumerate(parts):
        instance_id = i + 1
        mesh_name = os.path.splitext(part['file_name'])[0]
        instance_name = 'instance_%i'%(instance_id)
        part_count[mesh_name] += 1
        material_name = 'mat_%s'%part['color']
        #instance_mask_color = masks.index_to_mask_color(i)
        instance_mask_color = masks.color_index_to_float(instance_id)
        part_transform = numpy.dot(upright, part['transform'])
        renderpy_data['instances'][instance_name] = {
            'mesh_name' : mesh_name,
            'material_name' : material_name,
            'transform' : part_transform.tolist(),
            'mask_color' : instance_mask_color
        }
        x = part_transform[0,3]
        y = part_transform[1,3]
        z = part_transform[2,3]
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    
    center_x = range_x * 0.5 + min_x
    center_y = range_y * 0.5 + min_y
    center_z = range_z * 0.5 + min_z
    
    distance = max(range_x, range_y, range_z)*3
    
    camera_pose = camera.azimuthal_pose_to_matrix(
            [-0.6, -0.3, 0, distance, 0, 0])
    
    center_translation = numpy.eye(4)
    center_translation[0,3] -= center_x
    center_translation[1,3] -= center_y
    center_translation[2,3] -= center_z
    camera_pose = numpy.dot(camera_pose, center_translation)
    
    projection = camera.projection_matrix(
            math.radians(70), 1, 1, 5000).tolist()
    
    renderpy_data['camera'] = {
        'pose' : camera_pose,
        'projection' : projection
    }
    
    return renderpy_data
