#!/usr/bin/env python
import random

import numpy

from PIL import Image

import tqdm

import renderpy.masks as masks

from brick_gym.gym.brick_env import async_brick_env
from brick_gym.gym.standard_envs import segmentation_supervision_env

def generate_images(
        dataset = 'tiny_turbos',
        split = 'train',
        num_processes = 16):
    
    env = async_brick_env(
            num_processes,
            segmentation_supervision_env,
            dataset = dataset,
            split = split)
    
    env.reset()
    action = [{'visibility':0} for _ in range(num_processes)]
    for i in tqdm.tqdm(range(100)):
        observation, reward, terminal, info = env.step(action)
        action = []
        for j in range(len(observation['color_render'])):
            img = Image.fromarray(observation['color_render'][j])
            img.save('color_%06i.png'%(i*num_processes+j))
            seg = observation['segmentation_render'][j]
            mask = Image.fromarray(masks.color_index_to_byte(seg))
            mask.save('mask_%06i.png'%(i*num_processes+j))
            visible_indices = numpy.unique(seg)[1:] # don't choose 0
            a = {'visibility' : random.choice(visible_indices)}
            action.append(a)

if __name__ == '__main__':
    generate_images()
