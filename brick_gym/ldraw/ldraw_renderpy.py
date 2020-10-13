import math
import os

import numpy

import renderpy.camera as camera

import mpd
import colors

import brick_gym.config as config

upright = numpy.array([
    [-1, 0, 0, 0],
    [ 0,-1, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, 0, 0, 1]])

def mpd_to_renderpy(mpd_data,
        image_light_directory = None,
        ambient_color = (0,0,0)):
    
    obj_directory = config.paths['obj']
    external_parts = os.listdir(os.path.join(config.paths['ldraw'], 'parts'))
    
    parts = mpd.parts_from_mpd(mpd_data, external_parts)
    unique_dats = set(part['file_name'] for part in parts)
    unique_objs = [
            os.path.splitext(unique_dat)[0] + '.obj'
            for unique_dat in unique_dats]
    
    projection = camera.projection_matrix(
            math.radians(70), 1, 1, 5000).tolist()
    
    renderpy_data = {
        'image_lights' : {},
        'active_image_light' : 'background',
        'meshes': {},
        'materials' : {},
        'instances' : {},
        'ambient_color' : ambient_color,
        'point_lights' : {},
        'direction_lights' : {},
        'camera' : {
            'pose' : [0.0, -0.3, 0, 600, 0, 0],
            'projection' : projection
        }
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
        mesh_path = os.path.abspath(os.path.join(obj_directory, unique_obj))
        renderpy_data['meshes'][mesh_name] = {
            'mesh_path' : mesh_path,
            'scale' : 1.0,
            'create_uvs' : True
        }
    
    # materials
    unique_colors = set(part['color'] for part in parts)
    for unique_color in unique_colors:
        unique_color = int(unique_color)
        if unique_color in colors.colors:
            color = colors.colors[unique_color]
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
            'image_light_kd' : 0.8,
            'image_light_ks' : 0.2,
            'image_light_blur_reflection' : 3.0
        }
    
    # instances
    for i, part in enumerate(parts):
        mesh_name = os.path.splitext(part['file_name'])[0]
        instance_name = 'instance_%i'%i
        part_count[mesh_name] += 1
        material_name = 'mat_%s'%part['color']
        instance_mask_color = colors.index_to_color(i)
        renderpy_data['instances'][instance_name] = {
            'mesh_name' : mesh_name,
            'material_name' : material_name,
            'transform' : numpy.dot(upright, part['transform']).tolist(),
            'mask_color' : instance_mask_color
        }
    
    return renderpy_data
