import os
import json

import numpy

import gym.spaces

from ltron.gym.spaces import (
        ImageSpace, SegmentationSpace, StepSpace,
        PixelSelectionSpace)

import PIL.Image as Image

import splendor.masks as masks

def gym_dump(data, space, path, dump_json=True):
    if isinstance(space, ImageSpace):
        if len(data.shape) == 3:
            image_path = '%s.png'%path
            Image.fromarray(data).save(image_path)
            return os.path.basename(image_path)
        elif len(data.shape) == 4:
            result = []
            for i in range(data.shape[0]):
                result.append(gym_dump(
                        data[i], space, '%s_%02i'%(path, i), dump_json=False))
            return result
    
    elif isinstance(space, SegmentationSpace):
        if len(data.shape) == 2:
            image_path = '%s.png'%path
            colorized_data = masks.color_index_to_byte(data)
            Image.fromarray(colorized_data).save(image_path)
            return os.path.basename(image_path)
        elif len(data.shape) == 3:
            result = []
            for i in range(data.shape[0]):
                result.append(gym_dump(
                        data[i], space, '%s_%02i'%(path, i), dump_json=False))
            return result
        
    elif isinstance(space, StepSpace):
        pass
    elif isinstance(space, PixelSelectionSpace):
        pass
    elif isinstance(space, gym.spaces.Dict):
        json_data = {}
        for key, value in data.items():
            key_path = '%s_%s'%(path, key)
            key_data = gym_dump(value, space[key], key_path, dump_json=False)
            if key_data is not None:
                json_data[key] = key_data
        if dump_json:
            with open('%s.json'%path, 'w') as f:
                json.dump(json_data, f, indent=2)
    
    elif isinstance(space, gym.spaces.Tuple):
        json_data = []
        for i, value in enumerate(space):
            i_path = '%s_%04i'%(path, i)
            i_data = gym_dump(value, space[i], i_path, dump_json=False)
            if i_data is not None:
                json_data.append(i_data)
        if dump_json:
            with open('%s.json'%path, 'w') as f:
                json.dump(json_data, f)
    
    else:
        print('Unsupported data type:', type(space))
