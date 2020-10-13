#!/usr/bin/env python
import tqdm

import brick_gym.config as config
import brick_gym.random_stack.random_stack_dataset as random_stack_dataset

train_dataset = random_stack_dataset.RandomStackEdgeDataset(
        config.paths['random_stack'], 'train')

test_dataset = random_stack_dataset.RandomStackEdgeDataset(
        config.paths['random_stack'], 'test')

def bricks_to_tuple(brick_a, brick_b):
    return tuple(brick_a) + tuple(brick_b)

train_edges = set()
positive_train_edges = set()
test_edges = set()
positive_test_edges = set()

for dataset, edge_set, positive_edge_set in (
        (train_dataset, train_edges, positive_train_edges),
        (test_dataset, test_edges, positive_test_edges)):
    for bricks, edges in tqdm.tqdm(dataset):
        bricks_per_model = bricks.shape[-1]
        for i in range(bricks_per_model):
            for j in range(i, bricks_per_model):
                edge_a = bricks_to_tuple(
                        bricks[:,i].cpu().tolist(), bricks[:,j].cpu().tolist())
                edge_b = bricks_to_tuple(
                        bricks[:,j].cpu().tolist(), bricks[:,i].cpu().tolist())
                edge_set.add(edge_a)
                edge_set.add(edge_b)
                if int(edges[i,j]):
                    positive_edge_set.add(edge_a)
                    positive_edge_set.add(edge_b)

print('train edges: %i'%len(train_edges))
print('positive train edges: %i'%len(positive_train_edges))
print('test edges: %i'%len(test_edges))
print('positive test edges: %i'%len(positive_test_edges))

print('edges shared in train and test: %i'%len(train_edges & test_edges))
print('unique edges in test set: %i'%len(test_edges - train_edges))

print('positive edges shared in train and test: %i'%len(
        positive_train_edges & positive_test_edges))
print('positive unique edges in test set: %i'%len(
        positive_test_edges - positive_train_edges))
