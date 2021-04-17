#!/usr/bin/env python
import os

import numpy

import PIL.Image as Image

import tqdm

import brick_gym.config as config
import brick_gym.random_stack.dataset as random_stack_dataset

do_edge_stats = False
do_occlusion_stats = True

if do_edge_stats:
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
                            bricks[:,i].cpu().tolist(),
                            bricks[:,j].cpu().tolist())
                    edge_b = bricks_to_tuple(
                            bricks[:,j].cpu().tolist(),
                            bricks[:,i].cpu().tolist())
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

if do_occlusion_stats:
    train_render_directory = os.path.join(
            config.paths['random_stack'], 'train_render')

    occlusion_percentages = []
    running_average = 0.0
    iterate = tqdm.tqdm(os.listdir(train_render_directory)[:5000])
    for file_name in iterate:
        if file_name[-4:] != '.npy':
            continue
        file_path = os.path.join(train_render_directory, file_name)
        with open(file_path, 'rb') as f:
            mask_data = numpy.load(f)
        
        bricks_per_model = mask_data.shape[-1]//2
        
        for i in range(bricks_per_model):
            unoccluded_mask = mask_data[:,:,i*2+1]
            unoccluded_pixels = numpy.sum(unoccluded_mask)
            if not unoccluded_pixels:
                continue
            occluded_mask = mask_data[:,:,i*2]
            occluded_pixels = numpy.sum(occluded_mask)
            occlusion_percent = 1. - (occluded_pixels / unoccluded_pixels)
            occlusion_percentages.append(occlusion_percent)
            running_average = running_average * 0.99 + occlusion_percent * 0.01
            
        iterate.set_description('Avg: %.04f'%running_average)
    
    bins = bins = [0.05 * i for i in range(20)]
    bins.append(1.0001)
    bin_counts = []
    for low, high in zip(bins[:-1], bins[1:]):
        count = len([o for o in occlusion_percentages if o >= low and o < high])
        share = count / len(occlusion_percentages)
        print('%.02f-%.02f: %f'%(low, high, share))
    
    import matplotlib.pyplot as pyplot
    pyplot.hist(occlusion_percentages, bins=20, rwidth=0.8)
    pyplot.show()
