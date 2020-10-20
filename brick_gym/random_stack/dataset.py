import os

import numpy

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import tqdm

import PIL.Image as Image

brick_ids = {
    '3005.dat' : 1,
    '3004.dat' : 2,
    '3003.dat' : 3,
    '3002.dat' : 4,
    '3001.dat' : 5,
    '2456.dat' : 6}

def bricks_edges_from_model_files(
        model_directory,
        model_files,
        max_bricks_per_model):
    
    bricks = torch.zeros(
            len(model_files), 5, max_bricks_per_model,
            dtype=torch.long)
    edges = torch.zeros(
            len(model_files), max_bricks_per_model, max_bricks_per_model,
            dtype=torch.long)
    print('Loading model data from: %s'%model_directory)
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
                    bricks[i, 0, j] = brick_id
                    x, y, z = line_parts[2:5]
                    x = int(float(x)/10)
                    y = int(float(y)/-24)
                    z = int(float(z)/10)
                    o = int(float(line_parts[5]))
                    bricks[i, 1, j] = x
                    bricks[i, 2, j] = y
                    bricks[i, 3, j] = z
                    bricks[i, 4, j] = o
                    j += 1
                
                if line_parts[0] == '0' and line_parts[1] == 'EDGE':
                    # add an edge
                    first_brick, second_brick = line_parts[2].split(',')
                    first_brick = int(first_brick)
                    second_brick = int(second_brick)
                    edges[i, first_brick, second_brick] = 1
                    edges[i, second_brick, first_brick] = 1
    
    return bricks, edges

class RandomStackEdgeDataset(Dataset):
    def __init__(self, directory, split, max_bricks_per_model=8, subset=None):
        model_directory = os.path.join(directory, split)
        model_files = sorted(
                model_file for model_file in os.listdir(model_directory)
                if model_file[-4:] == '.mpd')
        if subset is not None:
            model_files = model_files[:subset]
        
        bricks, edges = bricks_edges_from_model_files(
                model_directory, model_files, max_bricks_per_model)
        
        self.bricks = bricks.cuda()
        self.edges = edges.cuda()
    
    def __getitem__(self, index):
        return self.bricks[index], self.edges[index]
    
    def __len__(self):
        return self.bricks.shape[0]

class RandomStackSegmentationDataset(Dataset):
    def __init__(self, directory, split, max_bricks_per_model=8, subset=None):
        self.max_bricks_per_model = max_bricks_per_model
        self.image_directory = os.path.join(directory, split + '_render')
        self.image_files = sorted(
                image_file for image_file in os.listdir(self.image_directory)
                if image_file[:5] == 'color')
        if subset is not None:
            self.image_files = self.image_files[:subset]
        
        '''
        self.mask_files = [
                image_file.replace('color_', 'mask_').replace('.png', '.npy')
                for image_file in self.image_files]
        '''
        
        model_directory = os.path.join(directory, split)
        model_files = list(sorted(set('model_%s.mpd'%(image_file.split('_')[1])
                for image_file in self.image_files)))
        
        model_file_index = dict(zip(model_files, range(len(model_files))))
        self.image_index_to_model_index = {
                i:model_file_index['model_%s.mpd'%(image_file.split('_')[1])]
                for i, image_file in enumerate(self.image_files)}
        
        self.bricks, self.edges = bricks_edges_from_model_files(
                model_directory, model_files, max_bricks_per_model)
    
    def __getitem__(self, index):
        # load image
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_directory, image_file)
        image = transforms.functional.to_tensor(Image.open(image_path))
        _, height, width = image.shape
        
        # load masks
        #mask_file = self.mask_files[index]
        masks = numpy.zeros(
                (height, width, self.max_bricks_per_model*2),
                dtype = numpy.uint8)
        for i in range(self.max_bricks_per_model*2):
            mask_path = image_path.replace('color', 'mask')
            mask_path = mask_path.replace('.png', '_%02i.png'%i)
            try:
                mask = numpy.array(Image.open(mask_path))
                masks[:,:,i] = mask
            except FileNotFoundError:
                break
        
        '''
        mask_path = os.path.join(self.image_directory, mask_file)
        masks = numpy.load(open(mask_path, 'rb'))
        '''
        target = torch.zeros((masks.shape[0], masks.shape[1]), dtype=torch.long)
        
        model_id = self.image_index_to_model_index[index]
        max_bricks_per_model = masks.shape[-1]//2
        for i in range(max_bricks_per_model):
            class_id = self.bricks[model_id, 0, i]
            target = target + class_id * masks[:,:,i*2]
        
        return image, target
    
    def __len__(self):
        return len(self.image_files)
