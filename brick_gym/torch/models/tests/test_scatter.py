#!/usr/bin/env python
import torch
from torch_scatter import scatter_max

conf = torch.rand(2,6,8)

f = torch.rand(2,1,6,8)

segmentation = torch.LongTensor([[
[ 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 1, 1, 0, 0, 0, 0, 0],
[ 0, 1, 1, 0, 2, 0, 0, 0],
[ 0, 0, 1, 0, 2, 2, 0, 0],
[ 0, 0, 0, 0, 0, 2, 0, 0],
[ 0, 0, 0, 3, 3, 3, 0, 0]],[

[ 0, 0, 0, 0, 4, 4, 0, 0],
[ 1, 0, 2, 2, 0, 4, 0, 0],
[ 1, 0, 0, 0, 0, 4, 0, 0],
[ 1, 0, 0, 0, 4, 4, 0, 0],
[ 0, 0, 0, 0, 4, 4, 0, 5],
[ 0, 0, 0, 0, 4, 0, 0, 5]]])

print(f)

print(conf)
print(segmentation.shape)

cluster_conf, argmax = scatter_max(
        conf.view(2,-1), segmentation.view(2,-1))

print(cluster_conf)
print(argmax)

good_clusters = torch.where(cluster_conf > 0.95)
print(good_clusters[0])

source_index = argmax[good_clusters]
print(argmax[good_clusters])

print(f.view(2,1,-1)[good_clusters[0],:,source_index])

