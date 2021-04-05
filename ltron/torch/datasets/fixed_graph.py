import torch
from torchvision.transforms.functional import to_tensor

import numpy

import PIL.Image as Image

import renderpy.masks as masks

from brick_gym.dataset.paths import (
        get_dataset_paths, get_dataset_info, get_metadata)

class FixedGraphDataset(torch.utils.data.Dataset):
    def __init__(self,
            dataset,
            split_name,
            subset=None,
            rank=0,
            size=1,
            transform=to_tensor):
        
        self.mask_paths = get_dataset_paths(
                dataset, split_name, subset, rank, size)
        self.dataset_info = get_dataset_info(dataset)
        self.transform = transform
    
    def __getitem__(self, index):
        mask_path = self.mask_paths[index]
        image = Image.open(mask_path)
        image_tensor = self.transform(image)
        c, h, w = image_tensor.shape
        mask = numpy.array(image)
        brick_indices = torch.LongTensor(masks.color_byte_to_index(mask))
        
        max_instances = self.dataset_info['max_instances_per_scene']
        metadata = get_metadata(mask_path)
        x = torch.zeros(max_instances, c, h, w)
        y_node = torch.zeros(max_instances, dtype=torch.long)
        for i in range(max_instances):
            brick_id = i+1
            if str(brick_id) in metadata['class_labels']:
                x[i] = image_tensor * (brick_indices == (brick_id))
                y_node[i] = metadata['class_labels'][str(brick_id)]
        
        all_edges = set(tuple(edge) for edge in metadata['edges'])
        y_edge = torch.zeros(max_instances, max_instances, dtype=torch.long)
        for i in range(max_instances):
            for j in range(i+1, max_instances):
                if (i+1, j+1) in all_edges:
                    y_edge[i,j] = 1
                    y_edge[j,i] = 1
        
        return x, y_node, y_edge
    
    def __len__(self):
        return len(self.mask_paths)
