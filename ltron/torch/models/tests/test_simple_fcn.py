#!/usr/bin/env python
import torch
import ltron.torch.models.simple_fcn as simple_fcn

fcn = simple_fcn.SimpleFCN()

img = torch.zeros(4, 3, 256, 256)

x = fcn(img)

print(x.shape)
