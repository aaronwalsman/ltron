import numpy

import torch
from torchvision.transforms.functional import to_tensor

def images_to_tensor(images, transform=to_tensor):
    bs, h, w, c = images.shape
    tensor = torch.stack(tuple(transform(image) for image in images))
    return tensor

def segmentations_to_tensor(segmentations):
    bs, h, w = segmentations.shape
    tensor = torch.stack(tuple(torch.LongTensor(seg) for seg in segmentations))
    return tensor

def images_masks_to_segment_tensor(
        images, masks, max_segments, transform=to_tensor):
    bs, h, w, c = images.shape
    tensor = torch.zeros(bs, max_segments, c+1, h, w)
    tensor[:,:,:c] = torch.stack(
            tuple(transform(im) for im in images)).unsqueeze(1)
    for i in range(max_segments):
        segment = torch.FloatTensor(masks == i+1).unsqueeze(1)
        tensor[:,i,c:] = segment
    
    return tensor

def image_segments_to_tensor(images, segments, transform=to_tensor):
    bs = len(images)
    h, w, c = images[0].shape
    s = len(segments[0])
    tensor = torch.zeros(bs, s, c+1, h, w)
    tensor[:,:,c:] = segments_to_tensor(segments)
    tensor[:,:,:c] = torch.stack(
            tuple(transform(im) for im in images)).unsqueeze(1)
    
    return tensor
