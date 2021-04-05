import os
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy

import PIL.Image as Image

import renderpy.masks as masks

from ltron.dataset.paths import data_paths

class SegmentationDataset(Dataset):
    def __init__(self,
            directory,
            split,
            subset=None,
            include_class_labels=True):
        
        self.image_paths = data_paths(directory, split, subset)
        self.include_class_labels = include_class_labels
        
        self.metadata_paths = []
        for image_path in self.image_paths:
            image_directory, image_file = os.path.split(image_path)
            model_index = image_file.split('_')[1]
            self.metadata_paths.append(os.path.join(
                    image_directory, 'metadata_%s.json'%model_index))
        
        if self.include_class_labels:
            self.class_ids = json.load(
                    open(os.path.join(directory, 'class_ids.json')))
            self.num_classes = max(self.class_ids.values()) + 1
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = transforms.functional.to_tensor(Image.open(image_path))
        
        mask_path = image_path.replace('color', 'mask')
        mask = numpy.array(Image.open(mask_path))
        instance_labels = torch.LongTensor(
                masks.color_byte_to_index(mask))
        
        if self.include_class_labels:
            image_directory, image_file = os.path.split(image_path)
            model_index = image_file.split('_')[1]
            metadata_file = 'metadata_%s.json'%model_index
            metadata_path = os.path.join(image_directory, metadata_file)
            metadata = json.load(open(metadata_path))
            class_labels = torch.LongTensor(
                    metadata['class_labels'])[instance_labels]
            
            return image, instance_labels, class_labels
        
        else:
            return image, instance_labels
    
    def __len__(self):
        return len(self.image_paths)
