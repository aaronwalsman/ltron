import math

import torch

EPS = 1e-5

class Offset2D(torch.nn.Module):
    def __init__(self, channels, downsample=1.0):
        super(Offset2D, self).__init__()
        # this conv is used to compute the offset and attention of each pixel
        self.offset_attention_conv = torch.nn.Conv2d(channels, 3, kernel_size=1)
        self.downsample = downsample
    
    def offset_sum_downsample(x, offset):
        # get the dimensions of the x tensor
        batch_size, channels, height, width = x.shape
        down_height = int(round(height * self.downsample))
        down_width = int(round(width * self.downsample))
        device = x.device
        
        offset[:,0] = offset[:,0] * down_height
        offset[:,1] = offset[:,1] * down_width
        
        destination = torch.zeros(batch_size, 2, height, width, device=device)
        destination[:,0] = torch.arange(height).unsqueeze(1).to(device) / height
        destination[:,1] = torch.arange(width).unsqueeze(0).to(device) / width
        destination = destination.to(x.device) + offset
        destination = torch.clamp(destination, 0.0, 1.0-EPS)
        destination[:,0] = torch.floor(destination[:,0] * down_height)
        destination[:,1] = torch.floor(destination[:,1] * down_width)
        destination = destination[:,0] * down_width + destination[:,1]
        destination = destination.unsqueeze(1).expand(
                batch_size, channels, height, width)
        downsample_batch_offset = (
                torch.arange(batch_size) * (channels*down_height*down_width))
        destination = destination + downsample_batch_offset.view(
                batch_size,1,1,1)
        downsample_channel_offset = (
                torch.arange(channels) * (down_height*down_width))
        destination = destination + downsample_channel_offset.view(channels,1,1)
        destination_unroll = destination.view(-1).long()
    
    def forward(self, x):
        # get the dimensions of the x tensor
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # compute the offset and attention values
        offset_attention = self.offset_attention_conv(x)
        
        offset = offset_attention[:,:2]
        
        attention = offset_attention[:,[2]]
        attention = torch.exp(attention)
        #attention = attention.reshape(batch_size, 1, height*width)
        
        '''
        destination = torch.arange(batch_size * channels * width * height)
        destination = destination.view(batch_size, channels, width, height)
        destination = destination + discrete_offset[:,[0]]
        destination = destination + discrete_offset[:,[1]] * width
        '''
        
        '''
        # use the offsets to generate the destination pixels
        destination_y = torch.arange(height, device=x.device)
        destination_y = destination_y.view(1,1,height,1)
        destination_y = destination_y.expand(batch_size,1,height,width)
        destination_x = torch.arange(width, device=x.device)
        destination_x = destination_x.view(1,1,1,width)
        destination_x = destination_x.expand(batch_size,1,height,width)
        destination = torch.cat((destination_y, destination_x), dim=1)
        destination = torch.round(destination + offset).long()
        #destination_unroll = destination.reshape(batch_size, 2, height*width)
        # index_add_ doesn't work quite how I hoped... may need to unroll all
        destination_unroll = 
        '''
        '''
        destination_y = torch.arange(height, device=x.device) / height
        destination_y = destination_y.view(1,1,height,1)
        destination_x = torch.arange(width, device=y.device) / width
        destination_x = destination_x.view(1,1,width,1)
        '''
        destination = torch.zeros(batch_size, 2, height, width, device=device)
        destination[:,0] = torch.arange(height).unsqueeze(1).to(device) / height
        destination[:,1] = torch.arange(width).unsqueeze(0).to(device) / width
        destination = destination.to(x.device) + offset
        destination = torch.clamp(destination, 0.0, 1.0-EPS)
        destination[:,0] = torch.floor(destination[:,0] * down_height)
        destination[:,1] = torch.floor(destination[:,1] * down_width)
        destination = destination[:,0] * down_width + destination[:,1]
        destination = destination.unsqueeze(1).expand(
                batch_size, channels, height, width)
        downsample_batch_offset = (
                torch.arange(batch_size) * (channels*down_height*down_width))
        destination = destination + downsample_batch_offset.view(
                batch_size,1,1,1)
        downsample_channel_offset = (
                torch.arange(channels) * (down_height*down_width))
        destination = destination + downsample_channel_offset.view(channels,1,1)
        destination_unroll = destination.view(-1).long()
        
        # accumulate x into the pixels specified by the destination tensor
        x = x * attention
        x = x.reshape(batch_size*channels*height*width)
        feature_accumulator = torch.zeros(
                batch_size*channels*down_height*down_width, device=x.device)
        feature_accumulator.index_add_(0, destination_unroll, x)
        
        # accumulate the attention into the pixels specified by the destination
        attention_accumulator = torch.zeros(
                (batch_size, 1, height*width), device=x.device).fill_(EPS)
        attention_accumulator.index_add_(-1, destination_unroll, attention)
        
        # normalize the feature accumulator and reshape back to height x width
        x = feature_accumulator / attention_accumulator
        x = x.reshape(batch_size, channels, down_height, down_width)
        
        return x, offset, destination

class FCNOffset2D(torch.nn.Module):
    def __init__(self, fcn, channels, classes, downsample=1.0):
        super(FCNOffset2D, self).__init__()
        self.fcn = fcn
        self.offset2d = Offset2D(channels, downsample=downsample)
        self.output_conv = torch.nn.Conv2d(channels, classes, kernel_size=1)
    
    def forward(self, x):
        x = self.fcn(x)
        x, offset, destination = self.offset2d(x)
        x = self.output_conv(x)
        
        return x, offset, destination

def offset_targets(index_map, indices):
    device = index_map.device
    targets = torch.zeros((2, *index_map.shape), device=device)
    centers = torch.zeros((len(indices), 2), device=device)
    valid_centers = torch.zeros(len(indices), dtype=torch.long, device=device)
    for i, index in enumerate(indices):
        yx = torch.nonzero(index_map == index, as_tuple=False)
        if not yx.shape[0]:
            continue
        
        hw = torch.FloatTensor(list(index_map.shape)).to(device)
        
        min_yx, _ = torch.min(yx, dim=0)
        max_yx, _ = torch.max(yx, dim=0)
        center_yx = (min_yx + max_yx).float() * 0.5
        centers[i] = center_yx / hw
        valid_centers[i] = 1
        
        offsets = (center_yx - yx.float())
        offsets = offsets / hw
        targets[:, yx[:,0], yx[:,1]] = offsets.transpose(0,1)
    
    return targets, centers, valid_centers
