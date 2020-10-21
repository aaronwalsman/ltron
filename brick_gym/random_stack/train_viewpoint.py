#!/usr/bin/env python
import torch
from torchvision.transforms.functional import to_tensor

import numpy

import PIL.Image as Image

import tqdm

import gym

import segmentation_models_pytorch

import brick_gym
import brick_gym.config as config
import brick_gym.masks as masks
import brick_gym.interactive.greedy as greedy

mode = 'train' # train/test

segmentation_model = segmentation_models_pytorch.FPN(
        encoder_name = 'se_resnext50_32x4d',
        encoder_weights = 'imagenet',
        classes = 7,
        activation = None).cuda()
checkpoint = (
        '../../runs/segmentation_train_001/segmentation_checkpoint_0010.pt')
state_dict = torch.load(checkpoint)
segmentation_model.load_state_dict(state_dict)

def reward_function(image, mask):
    image = to_tensor(image).unsqueeze(0).cuda()
    logits = segmentation_model(image)
    
    for i in range(7):
        mask = masks.get_mask
    with torch.no_grad():
        

if mode == 'train':
    train_env = gym.make('viewpoint-v0',
            directory = config.paths['random_stack'],
            split = 'train',
            reward_function = reward_function)
    
test_env = gym.make('viewpoint-v0',
        directory = config.paths['random_stack'],
        split = 'test',
        reward_function = reward_function)

class ViewpointModel(torch.nn.Module):
    def __init__(self, segmentation_model):
        super(ViewpointModel, self).__init__()
        self.segmentation_model = segmentation_model
        self.linear1 = torch.nn.Linear(2048, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 512)
        self.linear4 = torch.nn.Linear(512, 5)
    
    def forward(self, x):
        x = self.segmentation_model.encoder(x)[-1]
        x = torch.mean(x, dim=(2,3))
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        x = torch.nn.functional.relu(x)
        return self.linear4(x)

viewpoint_model = ViewpointModel(segmentation_model)

optimizer = torch.optim.Adam(viewpoint_model.parameters(), lr=3e-4)

num_epochs = 10

def test_epoch(epoch):
    with torch.no_grad():
        correct = 0
        total = 0
        correct_no_background = 0
        total_no_background = 0
        test_iterate = tqdm.tqdm(test_loader)
        for i, (images, targets) in enumerate(test_iterate):
            images = images.cuda()
            targets = targets.cuda()
            logits = model(images)
            prediction = torch.argmax(logits, dim=1)
            
            correct += torch.sum(prediction == targets)
            total += targets.numel()
            
            correct_no_background += torch.sum(
                    (prediction == targets) * (targets != 0))
            total_no_background += torch.sum(targets != 0)
            
            if i == 0:
                # make some images
                batch_size, _, height, width = images.shape
                target_mask_images = torch.zeros(
                        batch_size, 3, height, width, dtype=torch.uint8)
                predicted_mask_images = torch.zeros(
                        batch_size, 3, height, width, dtype=torch.uint8)
                for j in range(7):
                    mask_color = masks.color_floats_to_ints(
                            masks.index_to_mask_color(j))
                    mask_color = torch.ByteTensor(mask_color)
                    mask_color = (
                            mask_color.unsqueeze(0).unsqueeze(2).unsqueeze(3))
                    
                    target_mask_images += (
                            (targets.cpu() == j).unsqueeze(1) * mask_color)
                    predicted_mask_images += (
                            (prediction.cpu() == j).unsqueeze(1) * mask_color)
                
                for j in range(batch_size):
                    color_image = images[j].permute(1,2,0).cpu().numpy()
                    color_image = Image.fromarray(
                            (color_image * 255).astype(numpy.uint8))
                    color_image.save(
                            './color_%i_%i.png'%(epoch, j))
                    target_mask_image = Image.fromarray(
                            target_mask_images[j].permute(1,2,0).numpy())
                    target_mask_image.save(
                            './target_%i_%i.png'%(epoch, j))
                    predicted_mask_image = Image.fromarray(
                            predicted_mask_images[j].permute(1,2,0).numpy())
                    predicted_mask_image.save(
                            './predicted_%i_%i.png'%(epoch, j))
        
        print('Accuracy:               %f'%(float(correct)/total))
        print('Accuracy No Background: %f'%(
                float(correct_no_background)/float(total_no_background)))

if mode == 'train':
    for epoch in range(1, num_epochs+1):
        print('Epoch: %i'%epoch)
        print('Train')
        greedy.train_greedy(
                train_env,
                viewpoint_model,
                optimizer,
                num_actions = 5,
                num_batches = 128,
                batch_size = 16)
        
        checkpoint_path = './viewpoint_checkpoint_%04i.pt'%epoch
        print('Saving Checkpoint to: %s'%checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)
        
        #print('Test')
        #test_epoch(epoch)

elif mode == 'test':
    test_epoch(10)