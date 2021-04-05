import os
import json

import numpy

import gym.spaces

from ltron.gym.spaces import (
        ImageSpace, SegmentationSpace, StepSpace, SingleInstanceIndexSpace,
        PixelSelectionSpace, InstanceListSpace, EdgeSpace)
        #, EdgeScoreSpace,
        #SparseEdgeSpace, SparseEdgeScoreSpace, GraphScoreSpace,
        #SparseGraphScoreSpace)

import PIL.Image as Image

import renderpy.masks as masks
from renderpy.json_numpy import NumpyEncoder

def gym_log(label, data, space, log, step, log_json=True):
    if isinstance(space, ImageSpace):
        if len(data.shape) == 3:
            log.add_image(label, data, step, dataformats='HWC')
        elif len(data.shape) == 4:
            log.add_images(label, data, step, dataformats='NHWC')
        return label
    
    elif isinstance(space, SegmentationSpace):
        colorized_data = masks.color_index_to_byte(data)
        if len(data.shape) == 2:
            log.add_image(label, colorized_data, step, dataformats='HWC')
        elif len(data.shape) == 3:
            log.add_images(label, colorized_data, step, dataformats='NHWC')
        return label
        
    elif isinstance(space, StepSpace):
        pass
    elif isinstance(space, SingleInstanceIndexSpace):
        pass
    elif isinstance(space, PixelSelectionSpace):
        pass
    elif isinstance(space, EdgeSpace):
        return data
    elif isinstance(space, gym.spaces.Discrete):
        return data
    elif isinstance(space, gym.spaces.Box):
        return data
    elif isinstance(space, gym.spaces.Dict):
        json_data = {}
        for key, value in data.items():
            key_label = '%s/%s'%(label, key)
            key_data = gym_log(
                    key_label, value, space[key], log, step, log_json=False)
            if key_data is not None:
                json_data[key] = key_data
        if log_json:
            log.add_text(label, json.dumps(
                    json_data, cls=NumpyEncoder, indent=2), step)
    
    elif isinstance(space, gym.spaces.Tuple):
        json_data = []
        for i, value in enumerate(space):
            i_label = '%s/%04i'%(label, i)
            i_data = gym_log(
                    i_label, value, space[i], log, step, dump_json=False)
            if i_data is not None:
                json_data.append(i_data)
        if log_json:
            log.add_text(label, json.dumps(
                    json_data, cls=NumpyEncoder, indent=2), step)
    
    else:
        print('Unsupported data type:', type(space))
