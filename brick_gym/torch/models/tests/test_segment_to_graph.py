#!/usr/bin/env python
import torch
from brick_gym.torch.models.segmentation_to_graph import segmentation_to_graph

scores = torch.rand(2,6,8)

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

scores = scores * (segmentation != 0)

print(scores)
print(f)

graph_batch = segmentation_to_graph(scores, segmentation, {'f':f})
print(graph_batch.pos)
print(graph_batch.segment_index)
print(graph_batch.batch)
print(graph_batch.ptr)
