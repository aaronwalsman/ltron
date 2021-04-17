#!/usr/bin/env python
import torch
from torchvision.transforms.functional import to_tensor

import numpy

import PIL.Image as Image

import tqdm

from torch_scatter import scatter_add

from ltron.gym.standard_envs import graph_supervision_env
from ltron.torch.gym_tensor import (
        gym_space_to_tensors, gym_space_list_to_tensors, graph_to_gym_space)
from ltron.gym.brick_env import async_brick_env

if __name__ == '__main__':
    num_processes = 8
    train_dataset = 'rando_micro_wheels'
    train_split = 'all'
    train_subset = None
    segmentation_width, segmentation_height = 64, 64
    multi_hide = True
    max_instances_per_step = 8
    randomize_viewpoint = True
    randomize_colors = True

    train_env = async_brick_env(
                num_processes,
                graph_supervision_env,
                dataset=train_dataset,
                split=train_split,
                subset=train_subset,
                load_scenes=True,
                dataset_reset_mode='multi_pass',
                segmentation_width = segmentation_width,
                segmentation_height = segmentation_height,
                randomize_viewpoint=randomize_viewpoint,
                randomize_viewpoint_frequency='reset',
                randomize_colors=randomize_colors)

    num_steps = 1000
    
    instance_pixel_data = {}
    instance_z_data = {}
    
    step_observations = train_env.reset()
    for i in tqdm.tqdm(range(num_steps//num_processes)):
        step_tensors = gym_space_to_tensors(
                step_observations,
                train_env.single_observation_space,
                image_transform=to_tensor)
        
        seg = step_tensors['segmentation_render']
        
        instance_pixel_counts = scatter_add(
                torch.ones_like(seg).view(num_processes, -1),
                seg.view(num_processes, -1))
        
        for b in range(num_processes):
            save_frame = False
            for j in range(instance_pixel_counts.shape[-1]):
                if instance_pixel_counts[b,j] == 0:
                    continue
                instance_label = int((
                        step_tensors['graph_label'][b]['instance_label'][j]))
                if instance_label == 0:
                    continue
                
                if instance_label not in instance_pixel_data:
                    instance_pixel_data[instance_label] = []
                    instance_z_data[instance_label] = []
                
                instance_pixel_data[instance_label].append(
                        instance_pixel_counts[b,j])
                
                z = -step_tensors['brick_position']['camera'][b,j,2]
                instance_z_data[instance_label].append(z)
                
                '''
                if z > 700 or z < 200:
                    save_frame = True
                    if z > 700:
                        print('far', i, b)
                    else:
                        print('near', i, b)
                '''
            '''
            if save_frame:
                image = step_tensors['color_render'][b].cpu().numpy() * 255
                image = numpy.moveaxis(image,0,2)
                image = image.astype(numpy.uint8)
                Image.fromarray(image).save('tmp_%i_%i.png'%(i,b))
            '''
        
        step_observations = train_env.reset()
    
    print('Class: instance density, min p, max p, avg p, min z, max z, avg z')
    total_instances = sum([len(v) for v in instance_pixel_data.values()])
    min_all_z = float('inf')
    max_all_z = -float('inf')
    for class_label in sorted(instance_pixel_data.keys()):
        num_instances = len(instance_pixel_data[class_label])
        instance_density = num_instances/total_instances
        min_pixels = min(instance_pixel_data[class_label])
        max_pixels = max(instance_pixel_data[class_label])
        avg_pixels = sum(instance_pixel_data[class_label])/num_instances
        min_z = min(instance_z_data[class_label])
        max_z = max(instance_z_data[class_label])
        avg_z = sum(instance_z_data[class_label])/num_instances
        print('  %i: %.04f, %i, %i, %i, %.02f, %.02f, %.02f'%(
            class_label, instance_density,
            min_pixels, max_pixels, avg_pixels,
            min_z, max_z, avg_z))
        
        min_all_z = min(min_all_z, min_z)
        max_all_z = max(max_all_z, max_z)
    
    print('Min/Max Z: %.02f, %.02f'%(min_all_z, max_all_z))
