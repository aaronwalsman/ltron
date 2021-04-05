import os
import json

import torch
from torch.utils.data import Dataset

import tqdm

from ltron.dataset.paths import data_paths

def bricks_edges_from_model_path(model_path, max_bricks_per_model, class_ids):

    bricks = torch.zeros(5, max_bricks_per_model, dtype=torch.long)
    edges = torch.zeros(
            max_bricks_per_model, max_bricks_per_model, dtype=torch.long)

    with open(model_path) as f:
        j = 0
        for line in f.readlines():
            line_parts = line.split()
            if not len(line_parts):
                continue
            if line_parts[0] == '1':
                # add brick
                class_id = class_ids[line_parts[-1].strip()]
                bricks[0, j] = class_id
                x, y, z = line_parts[2:5]
                x = int(float(x)/10)
                y = int(float(y)/-24)
                z = int(float(z)/10)
                o = int(float(line_parts[5]))
                bricks[1, j] = x
                bricks[2, j] = y
                bricks[3, j] = z
                bricks[4, j] = o
                j += 1

            if line_parts[0] == '0' and line_parts[1] == 'EDGE':
                # add an edge
                first_brick, second_brick = line_parts[2].split(',')
                first_brick = int(first_brick)
                second_brick = int(second_brick)
                edges[first_brick, second_brick] = 1
                edges[second_brick, first_brick] = 1
    return bricks, edges

def bricks_edges_from_model_paths(model_paths, max_bricks_per_model, class_ids):

    bricks = torch.zeros(
            len(model_paths), 5, max_bricks_per_model,
            dtype=torch.long)
    edges = torch.zeros(
            len(model_paths), max_bricks_per_model, max_bricks_per_model,
            dtype=torch.long)
    for i, model_path in enumerate(tqdm.tqdm(model_paths)):
        bricks[i], edges[i] = bricks_edges_from_model_path(
                model_path, max_bricks_per_model, class_ids)

    return bricks, edges

class EdgeDataset(Dataset):
    def __init__(self, directory, split, subset=None, max_bricks_per_model=8):
        
        print('Loading model data from: %s'%directory)
        print('Loading split: %s'%split)
        model_paths = data_paths(directory, split, subset)
        class_ids = json.load(open(os.path.join(directory, 'class_ids.json')))
        bricks, edges = bricks_edges_from_model_paths(
                model_paths, max_bricks_per_model, class_ids)

        self.bricks = bricks.cuda()
        self.edges = edges.cuda()

    def __getitem__(self, index):
        return self.bricks[index], self.edges[index]

    def __len__(self):
        return self.bricks.shape[0]

