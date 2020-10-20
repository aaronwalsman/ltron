#!/usr/bin/env python
import torch

import numpy

import PIL.Image as Image

import tqdm

import segmentation_models_pytorch

import brick_gym.config as config
import brick_gym.ldraw.colors as colors
import brick_gym.random_stack.dataset as random_stack_dataset

mode = 'test' # train/test

if mode == 'train':
    train_dataset = random_stack_dataset.RandomStackSegmentationDataset(
            config.paths['random_stack'], 'train', subset=None)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True)

test_dataset = random_stack_dataset.RandomStackSegmentationDataset(
        config.paths['random_stack'], 'test', subset=None)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=True)

model = segmentation_models_pytorch.FPN(
        encoder_name = 'se_resnext50_32x4d',
        encoder_weights = 'imagenet',
        classes = 7,
        activation = None).cuda()

optim = torch.optim.Adam(model.parameters(), lr=3e-4)

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
                    mask_color = colors.color_floats_to_ints(
                            colors.index_to_mask_color(j))
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
        train_iterate = tqdm.tqdm(train_loader)
        for images, targets in train_iterate:
            images = images.cuda()
            targets = targets.cuda()
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            train_iterate.set_description('Loss: %.04f'%float(loss))
        
        checkpoint_path = './segmentation_checkpoint_%04i.pt'%epoch
        print('Saving Checkpoint to: %s'%checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)
        
        print('Test')
        test_epoch(epoch)

elif mode == 'test':
    checkpoint = (
            '../../runs/segmentation_train_001/segmentation_checkpoint_0010.pt')
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    test_epoch(10)
