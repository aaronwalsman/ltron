import os

import tqdm

import torch
from torch.utils.data import Dataset

brick_ids = {
    '3005.dat' : 1,
    '3004.dat' : 2,
    '3003.dat' : 3,
    '3002.dat' : 4,
    '3001.dat' : 5,
    '2456.dat' : 6}

class RandomStackEdgeDataset(Dataset):
    def __init__(self, directory, split, max_bricks_per_model=8, subset=None):
        model_directory = os.path.join(directory, split)
        model_files = sorted(
                model_file for model_file in os.listdir(model_directory)
                if model_file[-4:] == '.mpd')
        if subset:
            model_files = model_files[:subset]
        
        self.bricks = torch.zeros(
                len(model_files), 5, max_bricks_per_model,
                dtype=torch.long)
        self.edges = torch.zeros(
                len(model_files), max_bricks_per_model, max_bricks_per_model,
                dtype=torch.long)
        print('Loading %s data from: %s'%(split, directory))
        for i, model_file in enumerate(tqdm.tqdm(model_files)):
            with open(os.path.join(model_directory, model_file)) as f:
                j = 0
                for line in f.readlines():
                    line_parts = line.split()
                    if not len(line_parts):
                        continue
                    if line_parts[0] == '1':
                        # add brick
                        brick_id = brick_ids[line_parts[-1].strip()]
                        self.bricks[i, 0, j] = brick_id
                        x, y, z = line_parts[2:5]
                        x = int(float(x)/10)
                        y = int(float(y)/-24)
                        z = int(float(z)/10)
                        o = int(float(line_parts[5]))
                        self.bricks[i, 1, j] = x
                        self.bricks[i, 2, j] = y
                        self.bricks[i, 3, j] = z
                        self.bricks[i, 4, j] = o
                        j += 1
                    
                    if line_parts[0] == '0' and line_parts[1] == 'EDGE':
                        # add an edge
                        first_brick, second_brick = line_parts[2].split(',')
                        first_brick = int(first_brick)
                        second_brick = int(second_brick)
                        self.edges[i, first_brick, second_brick] = 1
                        self.edges[i, second_brick, first_brick] = 1
        
        self.bricks = self.bricks.cuda()
        self.edges = self.edges.cuda()
    
    def __getitem__(self, index):
        return self.bricks[index], self.edges[index]
    
    def __len__(self):
        return self.bricks.shape[0]
