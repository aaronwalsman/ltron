#!/usr/bin/env python
import torch
import ltron.model.offset2d as offset2d

layer = offset2d.Offset2D(4, 0.5)

height = 8
width = 8

x = torch.zeros(2, 4, height, width)
x[:,:] = torch.arange(4).unsqueeze(-1).unsqueeze(-1)

offset1 = torch.zeros(2,2,height,width)
offset1_y = (5 - torch.arange(height))
offset1_x = (0 - torch.arange(width))
offset1[0,0] = offset1_y.unsqueeze(-1) / float(height)
offset1[0,1] = offset1_x.unsqueeze(0) / float(width)

offset2 = torch.zeros(2,2,height,width)
offset2_y = (2 - torch.arange(height))
offset2_x = (7 - torch.arange(width))
offset2[0,0] = offset2_y.unsqueeze(-1) / float(height)
offset2[0,1] = offset2_x.unsqueeze(0) / float(width)

offset = torch.cat([offset1[:,:,:,:5], offset2[:,:,:,5:]], dim=-1)

x_down, dest = layer.offset_sum_downsample(x, offset)

print(x_down)
print(dest)
