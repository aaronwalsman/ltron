#!/usr/bin/env python
import brick_gym.torch.train.graph as graph

def run():
    graph.alternate_reinforce_supervise(
            num_epochs = 1000,
            num_processes = 4,
            train_episodes_per_epoch = 256,
            test_episodes_per_epoch = 64,
            mini_epochs = 4,
            dataset = 'random_stack',
            node_model_name = 'spatial_resnet_18',
            edge_model_name = 'simple_edge_512',
            discount = 0.99)
