import numpy

import gym.spaces

from brick_gym.spaces import (
        ImageSpace, SegmentationSpace, StepSpace, InstanceSelectionSpace,
        PixelSelectionSpace, NodeSpace, EdgeScoreSpace, SparseEdgeSpace,
        SparseEdgeScoreSpace, GraphScoreSpace, SparseGraphScoreSpace)

import PIL.Image as Image

def save_gym_data(data, space, path):
    if isinstance(space, ImageSpace):
        if len(data.shape) == 3:
            image_path = '%s.png'%path
            Image.fromarray(data).save(image_path)
        elif len(data.shape) == 4:
            for i in range(data.shape[0]):
                image_path = '%s_%02i.png'%(path, i)
                Image.fromarray(data[i]).save(image_path)
    elif isinstance(space, SegmentationSpace):
        max_instances = numpy.max(data)
        def save_segmentations(d, image_path):
            im = (d == j+1).astype(numpy.uint8) * 255
            Image.fromarray(im).save(image_path)
        if len(data.shape) == 2:
            for j in range(max_instances):
                image_path = '%s_%04i.png'%(path, j)
                save_segmentations(data, image_path)
        elif len(data.shape) == 3:
            for i in range(data.shape[0]):
                for j in range(max_instances):
                    image_path = '%s_%02i_%04i.png'%(path, i, j)
                    save_segmentations(data[i], image_path)
        
    elif isinstance(space, StepSpace):
        pass
    elif isinstance(space, InstanceSelectionSpace):
        pass
    elif isinstance(space, PixelSelectionSpace):
        pass
    elif isinstance(space, NodeSpace):
        pass
    elif isinstance(space, EdgeScoreSpace):
        pass
    elif isinstance(space, SparseEdgeSpace):
        pass
    elif isinstance(space, SparseEdgeScoreSpace):
        pass
    elif isinstance(space, GraphScoreSpace):
        pass
    elif isinstance(space, SparseGraphScoreSpace):
        pass
    elif isinstance(space, gym.spaces.Dict):
        for key, value in data.items():
            key_path = '%s_%s'%(path, key)
            save_gym_data(value, space[key], key_path)
    elif isinstance(space, gym.spaces.Tuple):
        for i, value in enumerate(space):
            i_path = '%s_%04i'%(path, i)
            save_gym_data(value, space[i], i_path)
    else:
        print('Unsupported data type:', type(space))
