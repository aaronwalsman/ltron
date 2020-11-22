import math

import torch

EPS = 1e-5

class Offset2D(torch.nn.Module):
    def __init__(self, channels, downsample=1.0):
        super(Offset2D, self).__init__()
        # this conv is used to compute the offset and attention of each pixel
        self.offset_attention_conv = torch.nn.Conv2d(channels, 3, kernel_size=1)
        self.downsample = downsample
    
    def forward(self, x):
        # get the dimensions of the x tensor
        batch_size, channels, height, width = x.shape
        downsample_height = math.round(height * self.downsample)
        downsample_width = math.round(width * self.downsample)
        
        # compute the offset and attention values
        offset_attention = self.offset_attention_conv(x)
        
        offset = offset_attention[:,:2]
        offset[:,0] = offset[:,0] * downsample_height
        offset[:,1] = offset[:,1] * downsample_width
        
        attention = offset_attention[:,[2]]
        attention = torch.exp(attention)
        attention = attention.reshape(batch_size, 1, height*width)
        
        # use the offsets to generate the destination pixels
        destination_y = torch.arange(height, device=x.device)
        destination_y = destination.view(1,1,height,1)
        destination_y = destination_y.expand(batch_size,1,height,width)
        destination_x = torch.arange(width, device=x.device)
        destination_x = destination_x.view(1,1,1,width)
        destination_x = destination_x.expand(batch_size,1,height,width)
        destination = torch.cat((destination_y, destination_x), dim=1)
        destination = torch.round(destination + offset).long()
        destination_unroll = destination.reshape(batch_size, 2, height*width)
        
        # accumulate x into the pixels specified by the destination tensor
        x = x.reshape(batch_size, channels, height*width) * attention
        feature_accumulator = torch.zeros(x.shape, device=x.device)
        feature_accumulator.index_add_(-1, destination_unroll, x)
        
        # accumulate the attentino into the pixels specified by the destination
        attention_accumulator = torch.zeros(
                (batch_size, 1, height*width), device=x.device).fill_(EPS)
        attention_accumulator.index_add_(-1, destination_unroll, attention)
        
        # normalize the feature accumulator and reshape back to height x width
        x = feature_accumulator / attention_accumulator
        x = x.reshape(batch_size, channels, height, width)
        
        return x, offset, destination

class Offset2DSegmentationModel(torch.nn.Module):
    def __init__(fcn, channels, classes):
        super(Offset2DSegmentationModel, self).__init__()
        self.fcn = fcn
        self.offset2d = Offset2D(channels)
        self.output_conv = torch.nn.Conv2d(channels, classes, kernel_size=1)
    
    def forward(self, x):
        x = self.fcn(x)
        x, offset, destination = self.offset2d(x)
        x = self.output_conv(x)
        
        return x, offset, destination

def offset_targets(index_map, indices):
    device = index_map.device
    targets = torch.zeros((2, *index_map.shape), device=device)
    for index in indices:
        yx = torch.nonzero(index_map == index, as_tuple=False)
        if not yx.shape[0]:
            continue
        
        min_yx, _ = torch.min(yx, dim=0)
        max_yx, _ = torch.max(yx, dim=0)
        center_yx = (min_yx + max_yx).float() * 0.5
        
        offsets = (center_yx - yx.float())
        offsets = offsets / torch.FloatTensor(list(index_map.shape)).to(device)
        targets[:, yx[:,0], yx[:,1]] = offsets.transpose(0,1)
    
    return targets
