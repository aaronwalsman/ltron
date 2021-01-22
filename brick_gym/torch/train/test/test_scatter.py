#!/usr/bin/env python
import torch

g = torch.LongTensor([[
[0, 0, 0, 0],
[0, 1, 2, 1],
[0, 1, 1, 1],
[3, 0, 0, 0]],[
[1, 1, 0, 0],
[1, 1, 0, 3],
[1, 0, 0, 3],
[0, 0, 0, 3]]])

remap = torch.LongTensor([[0,5,3,7], [0,2,7,9]])

#print(torch.scatter(1, remap, g.view(2,-1)))

#print(remap[g.view(2,-1)])

a = torch.stack([remap[i][g[i]] for i in range(2)])
print(a)

a = remap[g]
print(a)
