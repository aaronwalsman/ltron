import math

import torch

EPS = 1e-5

class DiscreteOffset2D(torch.nn.Module):
    def __init__(self, channels, height, width, y_bins=None, x_bins=None):
        super(DiscreteOffset2D, self).__init__()
        self.height = height
        self.width = width
        if y_bins is None:
            y_bins = height * 2 - 1
        if x_bins is None:
            x_bins = width * 2 - 1
        self.y_bins = y_bins
        self.x_bins = x_bins
        
        self.offset_attention_conv = torch.nn.conv2d(
                channels, self.y_bins + self.x_bins + 1, kernel_size=1)
    
    def offset_sum_downsample(self, x, offset_y, offset_x):
        batch_size, channels, height, width = x.shape
        
    
    def forward(x):
        offset_attention = self.offset_attention_conv(x)
        offset_y = offset_attention[:,:self.y_bins]
        offset_x = offset_attention[:,self.y_bins:-2]
        attention = offset_attention[-1]
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Doing an outer product of the softmax of clipped
        # offset_y and offset_x would give us a distribution over all bins
        # for each pixel.  While this would be nice and differentiable,
        # this is probably waaaaaaaaay too many operations (HxWxY_BINxX_BIN).
        # Is there a convolution trick to do it with fewer?  I don't think so
        # because my kernels vary (every pixel produces it's own distribution
        # over bins).
        
        # So maybe I just take the argmax of y and x and just supervise like I
        # was doing with the real-valued offsets before?  It would be nice if
        # this was all fully differentiable though.
        
        # Another option would be to reduce the "reach" of each pixel (reduce
        # the number of bins it can map to).  This means we can't have big
        # segments, but we could then aggregate in stages?
        
        # The bigger question though is whether or not I'm off in the weeds at
        # this point.  Aren't I supposed to be working on Legos here?  Shouldn't
        # I be looking for some off-the-shelf solution to this?

class Offset2D(torch.nn.Module):
    def __init__(self, channels, downsample=1.0):
        super(Offset2D, self).__init__()
        # this conv is used to compute the offset and attention of each pixel
        self.offset_attention_conv = torch.nn.Conv2d(channels, 3, kernel_size=1)
        self.downsample = downsample
    
    def offset_sum_downsample(self, x, offset):
        # get the dimensions of the x tensor
        batch_size, channels, height, width = x.shape
        fheight = float(height)
        fwidth = float(width)
        down_height = int(round(height * self.downsample))
        down_width = int(round(width * self.downsample))
        device = x.device
        
        # build the destination tensor
        # each value in this tensor will store the unrolled destination
        # in the unrolled downsampled feature map of each pixel in source
        destination = torch.zeros(batch_size, 2, height, width, device=device)
        destination[:,0] = torch.arange(height).unsqueeze(1).to(device)/fheight
        destination[:,1] = torch.arange(width).unsqueeze(0).to(device)/fwidth
        # at this point destination stores the normalized initial locations
        destination = destination.to(device) + offset
        destination = torch.clamp(destination, 0.0, 1.0-EPS)
        # at this point destination stores the normalized offset locations
        destination[:,0] = torch.floor(destination[:,0] * down_height)
        destination[:,1] = torch.floor(destination[:,1] * down_width)
        destination = return_destination = destination.long()
        # at this point destination stores the unnormalized offset locations
        destination = destination[:,[0]] * down_width + destination[:,[1]]
        # at this point the values in destination have been unrolled in y and x
        down_batch_offset = torch.arange(batch_size, device=device).long()
        down_batch_offset *= channels*down_height*down_width
        down_channel_offset = torch.arange(channels, device=device).long()
        down_channel_offset *= down_height*down_width
        destination = destination + down_batch_offset.view(batch_size,1,1,1)
        destination = destination + down_channel_offset.view(channels,1,1)
        # at this point the values in destination have been unrolled in bs and c
        
        # accumulate x into the pixels specified by the destination tensor
        x = x.reshape(batch_size*channels*height*width)
        down_elements = batch_size*channels*down_height*down_width
        down_x = torch.zeros(down_elements, device=x.device)
        down_x.index_add_(0, destination.view(-1), x)
        down_x = down_x.view(batch_size, channels, down_height, down_width)
        
        return down_x, return_destination
    
    def forward(self, x):
        # get the dimensions of the x tensor
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # compute the offset and attention values
        offset_attention = self.offset_attention_conv(x)
        
        offset = offset_attention[:,:2]
        
        attention = offset_attention[:,[2]]
        #exp_attention = torch.exp(attention)
        exp_attention = torch.ones(batch_size, 1, height, width).to(x.device)
        
        # apply the attention values to x, and concate them to x
        # the concatenated version will accumulate the total attention
        x = x * exp_attention
        x_attention = torch.cat([x, exp_attention], dim=1)
        
        # compute the destination and accumulate
        x_attention, destination = self.offset_sum_downsample(
                x_attention, offset)
        
        # normalize x using the accumulated attention
        #x = x_attention[:,:-1] / (x_attention[:,[-1]] + EPS)
        x = x_attention
        x[:,-1] /= height * width
        
        return x, offset, attention, destination

class FCNOffset2D(torch.nn.Module):
    def __init__(self, fcn, channels, final_layers, classes, downsample=1.0):
        super(FCNOffset2D, self).__init__()
        self.fcn = fcn
        with torch.no_grad():
            test_image = torch.zeros(1,3,64,64)
            x = fcn(test_image)
            fcn_channels = x.shape[1]
        self.offset2d = Offset2D(fcn_channels, downsample=downsample)
        layers = []
        in_channels = fcn_channels + 1
        for i in range(final_layers):
            layers.append(torch.nn.Conv2d(in_channels, channels, kernel_size=1))
            layers.append(torch.nn.ReLU())
            in_channels = channels
        self.final_layers = torch.nn.Sequential(*layers)
        self.output_conv = torch.nn.Conv2d(channels, classes, kernel_size=1)
    
    def forward(self, x):
        x = self.fcn(x)
        x, offset, attention, destination = self.offset2d(x)
        x = self.final_layers(x)
        x = self.output_conv(x)
        
        return x, offset, attention, destination

def offset_targets(index_map, indices):
    device = index_map.device
    targets = torch.zeros((2, *index_map.shape), device=device)
    centers = torch.zeros((len(indices), 2), device=device)
    center_indices = torch.zeros(len(indices), dtype=torch.long, device=device)
    for i, index in enumerate(indices):
        yx = torch.nonzero(index_map == index, as_tuple=False)
        if not yx.shape[0]:
            continue
        
        hw = torch.FloatTensor(list(index_map.shape)).to(device)
        
        min_yx, _ = torch.min(yx, dim=0)
        max_yx, _ = torch.max(yx, dim=0)
        center_yx = (min_yx + max_yx).float() * 0.5
        centers[i] = center_yx / hw
        center_indices[i] = index
        
        offsets = (center_yx - yx.float())
        offsets = offsets / hw
        targets[:, yx[:,0], yx[:,1]] = offsets.transpose(0,1)
    
    return targets, centers, center_indices
