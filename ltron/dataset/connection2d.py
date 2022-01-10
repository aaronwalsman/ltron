import random
import os
import json

import torch

import numpy

import PIL.Image as Image

import tqdm

import splendor.masks as masks

import ltron.settings as settings

shapes = [
    (1,1),
    (2,1),
    (2,2),
    (4,1),
    (1,2),
    (1,4)
]

class_labels = {
    (1,1) : 1,
    (2,1) : 2,
    (2,2) : 3,
    (4,1) : 4,
    (1,2) : 5,
    (1,4) : 6
}

random.seed(1234)

def sample_configuration(height, width, min_bricks, max_bricks):
    occupancy = numpy.zeros((height, width), dtype=numpy.long)
    num_bricks = random.randint(min_bricks, max_bricks)
    bricks = []
    for i in range(1, num_bricks+1):
        shape = random.choice(shapes)
        for attempt in range(1000):
            y = random.randint(0, height-1)
            x = random.randint(0, width-1)
            if y + shape[0] > height:
                continue
            if x + shape[1] > width:
                continue
            if not numpy.sum(occupancy[y:y+shape[0], x:x+shape[1]]):
                bricks.append((shape, (y,x)))
                occupancy[y:y+shape[0], x:x+shape[1]] = i
                break
    
    edges = []
    for a in range(len(bricks)):
        (sy, sx), (y, x) = bricks[a]
        min_y = max(y-1, 0)
        min_x = max(x-1, 0)
        max_y = min(y+sy+1, height)
        max_x = min(x+sx+1, width)
        neighborhood = occupancy[min_y:max_y, min_x:max_x]
        neighbors = set(neighborhood.ravel().tolist()) - {0, a+1}
        for neighbor in neighbors:
            if neighbor > a+1:
                edges.append((a+1, neighbor))
    
    return bricks, edges

def render(bricks, height, width, step_y, step_x):
    image = numpy.zeros((height * step_y, width * step_x, 3), dtype=numpy.uint8)
    for i, brick in enumerate(bricks):
        brick_id = i+1
        (sy, sx), (y, x) = brick
        mask_color = masks.color_index_to_byte(brick_id)
        image[y*step_y:(y+sy)*step_y, x*step_x:(x+sx)*step_x] = mask_color
    
    return image

def make_connection2d_dataset(
        num_train=10000,
        num_test=2000,
        height=8,
        width=8,
        image_height=256,
        image_width=256,
        min_bricks=4,
        max_bricks=8):
    
    step_y = image_height // height
    step_x = image_width // width
    
    dataset_path = settings.datasets['connection2d']
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    for path in dataset_path, train_path, test_path:
        if not os.path.exists(path):
            os.makedirs(path)
    
    for num_examples, path, split in (
            (num_train, train_path, 'train'), (num_test, test_path, 'test')):
        print('Making %s'%split)
        for i in tqdm.tqdm(range(num_examples)):
            bricks, edges = sample_configuration(
                    height, width, min_bricks, max_bricks)
            image = render(bricks, height, width, step_y, step_x)
            image_path = os.path.join(path, 'mask_%06i.png'%i)
            Image.fromarray(image).save(image_path)
            
            metadata = {}
            metadata['class_labels'] = {
                    i+1 : class_labels[shape]
                    for i, (shape, _) in enumerate(bricks)}
            metadata['edges'] = edges
            metadata_path = os.path.join(path, 'metadata_%06i.json'%i)
            json.dump(metadata, open(metadata_path, 'w'))
            
    dataset_info = {}
    dataset_info['splits'] = {
            'train' : ['train/mask*.png'],
            'test' : ['test/mask*.png']
    }
    dataset_info['max_instances_per_scene'] = max_bricks
    dataset_info['shape_ids'] = {
            str(shape) : value
            for shape, value in class_labels.items()
    }
    dataset_info_path = os.path.join(dataset_path, 'dataset.json')
    json.dump(dataset_info, open(dataset_info_path, 'w'))
