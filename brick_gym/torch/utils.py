import torch
from torchvision.transforms.functional import to_tensor

def image_segments_to_tensor(images, segments, transform=to_tensor):
    '''
    ims = torch.stack(tuple(
    tensor = torch.stack(tuple(torch.stack(tuple(
                torch.cat((im, transform(seg)), dim=0) for seg in segs))
                for segs in segments))
    '''
    
    bs = len(images)
    h, w, c = images[0].shape
    s = len(segments[0])
    tensor = torch.zeros(bs, s, c*2, h, w)
    tensor[:,:,c:] = segments_to_tensor(segments)
    tensor[:,:,:c] = torch.stack(
            tuple(transform(im) for im in images)).unsqueeze(1)
    
    return tensor

def segments_to_tensor(segments, transform=to_tensor):
    tensor = torch.stack(tuple(torch.stack(tuple(
                transform(seg) for seg in segs))
                for segs in segments))
    return tensor
