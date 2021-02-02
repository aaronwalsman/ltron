#!/usr/bin/env python

import torch

seg = torch.FloatTensor(
[  [[0, 0, 0, 0],
    [0, 1, 1, 0],
    [1, 1, 2, 2],
    [0, 0, 0, 2]],
   [[0, 0, 1, 1],
    [2, 0, 3, 3],
    [2, 2, 3, 3],
    [2, 3, 3, 0]]]).view(2, -1)

scr = torch.FloatTensor(
[  [[0., 0., 0., 0.],
    [0., .1, .2, 0.],
    [.1, .1, .4, .5],
    [0., 0., 0., .4]],
   [[0., 0., .1, .8],
    [.8, 0., .9, .2],
    [.4, .2, .8, .1],
    [.3, .1, .1, 0.]]]).view(2, -1)

b = 2
max_seg = 4
locations = torch.zeros(b, max_seg)
for m in range(4):
    argm = torch.argmax(scr.view(b, -1), dim=-1)
    locations[:,m] = argm
    print(argm)
    s = seg[range(b),argm]
    scr = scr * (seg != s.view(-1, 1))
    #print((seg == s.view(-1, 1)).view(2,4,4))
    #print(scr.view(2,4,4))
