#!/usr/bin/env python
import brick_gym.torch.train.fixed_graph as fixed_graph

fixed_graph.train_fixed_graph(
        num_epochs = 10,
        dataset = 'connection2d',
        node_model_name = 'spatial_resnet_18',
        edge_model_name = 'simple_edge_512')
