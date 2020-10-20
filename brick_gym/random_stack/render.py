#!/usr/bin/env python
import time
import random
import math
import os

import numpy

import PIL.Image as Image

import tqdm

import renderpy.buffer_manager_glut as buffer_manager
import renderpy.core as core
import renderpy.camera as camera

import brick_gym.config as config
import brick_gym.ldraw.colors as colors
import brick_gym.ldraw.ldraw_renderpy as ldraw_renderpy

default_image_light = '/home/awalsman/Development/renderpy/renderpy/example_image_lights/grey_cube'

def render_random_stack_dataset(
        directory,
        split,
        cameras_per_model = 4,
        max_bricks_per_model = 8,
        width = 256,
        height = 256,
        elevation_range = [-1.0, 1.0],
        subset=None):
    
    model_directory = os.path.join(directory, split)
    model_files = sorted(
            model_file for model_file in os.listdir(model_directory)
            if model_file[-4:] == '.mpd')
    
    output_directory = os.path.join(directory, split + '_render')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    manager = buffer_manager.initialize_shared_buffer_manager(width, height)
    renderer = core.Renderpy()
    manager.add_frame('color', width, height, anti_aliasing=True)
    manager.add_frame('mask', width, height, anti_aliasing=False)
    
    for i, model_file in enumerate(tqdm.tqdm(model_files)):
        with open(os.path.join(model_directory, model_file)) as f:
            scene = ldraw_renderpy.mpd_to_renderpy(
                    f, image_light_directory = default_image_light)
        
        if i == 0:
            renderer.load_scene(scene, clear_existing=True)
        else:
            renderer.clear_instances()
            for mesh, mesh_data in scene['meshes'].items():
                if not renderer.mesh_exists(mesh):
                    renderer.load_mesh(mesh, **mesh_data)
            renderer.load_scene(
                    {'instances':scene['instances']}, clear_existing = False)
        bbox_min, bbox_max = renderer.get_instance_center_bbox()
        bbox_range = bbox_max - bbox_min
        scene_center = bbox_min + bbox_range * 0.5
        camera_distance = numpy.max(bbox_range) * 3
        
        camera_transforms = [
                camera.azimuthal_pose_to_matrix([
                    random.uniform(0, math.pi*2),
                    random.uniform(*elevation_range),
                    0, camera_distance, 0, 0], scene_center)
                for _ in range(cameras_per_model)]
        for j, camera_transform in enumerate(camera_transforms):
        
            renderer.set_camera_pose(camera_transform)
            
            manager.enable_frame('color')
            renderer.color_render()
            color_image = manager.read_pixels('color')
            color_path = os.path.join(
                    output_directory, 'color_%06i_%04i.png'%(i,j))
            color_image = Image.fromarray(color_image)
            color_image.save(color_path)
            '''
            mask_data = numpy.zeros(
                    (height, width, max_bricks_per_model*2),
                    dtype=numpy.uint8)
            '''
            manager.enable_frame('mask')
            renderer.mask_render()
            occluded_mask = manager.read_pixels('mask')
            
            for instance_name, instance_data in scene['instances'].items():
                instance_id = int(instance_name.split('_')[-1])
                mask_color = instance_data['mask_color']
                mask_color = colors.color_floats_to_ints(mask_color)
                mask = colors.get_mask(occluded_mask, mask_color)
                mask_path = os.path.join(
                        output_directory,
                        'mask_%06i_%04i_%02i.png'%(i,j,instance_id*2))
                mask_image = Image.fromarray(mask)
                mask_image.save(mask_path)
                #mask_data[:,:,instance_id*2] = mask
                
                renderer.mask_render([instance_name])
                unoccluded_mask = manager.read_pixels('mask')
                mask = colors.get_mask(unoccluded_mask, mask_color)
                mask_path = os.path.join(
                        output_directory,
                        'mask_%06i_%04i_%02i.png'%(i,j,instance_id*2+1))
                mask_image = Image.fromarray(mask)
                mask_image.save(mask_path)
                #mask_data[:,:,instance_id*2+1] = mask
            #mask_path = os.path.join(
            #        output_directory, 'mask_%06i_%04i.npy'%(i,j))
            #with open(mask_path, 'wb') as f:
            #    numpy.save(f, mask_data)

if __name__ == '__main__':
    render_random_stack_dataset(
            config.paths['random_stack'],
            'train')
