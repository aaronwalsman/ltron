#!/usr/bin/env python
import torch

from brick_gym.torch.brick_graph import BrickList, BrickGraph

'''
0-1
| |
2-3
|/
4
'''
a_list = BrickList(labels=torch.LongTensor([0,1,2,3,4]).cuda())
a_graph = BrickGraph(a_list, torch.LongTensor([
        [0,1,0,2,2,1,3,3,2,4,3,4],
        [1,0,2,0,3,3,1,2,4,2,4,3]]))

'''
5-6-7-8-9
'''
b_list = BrickList(labels=torch.LongTensor([5,6,7,8,9]).cuda())
b_graph = BrickGraph(b_list, torch.LongTensor([
        [0,1,1,2,2,3,3,4],
        [1,0,2,1,3,2,4,3]]))

matching_nodes = torch.LongTensor([
        [0,3],
        [1,0]])
new_edges = torch.LongTensor([
        [1,2],
        [3,3]])
a_graph.merge(b_graph, matching_nodes=matching_nodes, new_edges=new_edges)
'''
  +-----+
  |     |
9-8-5-6-7
  | |/  |
  2-3---+
  |/
  4
'''
print(a_graph.labels)
print(a_graph.edge_index)
