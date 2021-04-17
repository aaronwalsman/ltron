#!/usr/bin/env python
import tqdm

import torch

from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv

from ltron.torch.models.edge import EdgeModel

graph_conv = GCNConv(32, 1)
edge_model = EdgeModel(32, graph_conv).cuda()

for i in tqdm.tqdm(range(100)):
    x_a = torch.rand(2000, 32).cuda()
    x_b = torch.rand(18, 32).cuda()

    graph_a = GraphData(x_a)
    graph_b = GraphData(x_b)

    edge_logits = edge_model(graph_a, graph_b)

