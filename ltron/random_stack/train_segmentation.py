#!/usr/bin/env python
import argparse

import torch

import numpy

import PIL.Image as Image

import tqdm

import segmentation_models_pytorch

import renderpy.masks as masks

import brick_gym.config as config
import brick_gym.random_stack.dataset as random_stack_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
        '--model', type=str, default='resnet18')
parser.add_argument(
        '--batch-size', type=int, default=64)
parser.add_argument(
        '--test', action='store_true')
parser.add_argument(
        '--num-epochs', type=int, default=25)
parser.add_argument(
        '--train-subset', type=int, default=None)
parser.add_argument(
        '--test-subset', type=int, default=None)
parser.add_argument(
        '--encoder-weights', type=str, default=None)
parser.add_argument(
        '--include-confidence', action='store_true')

args = parser.parse_args()

if args.test:
    mode = 'test'
else:
    mode = 'train'

model_type = args.model #'resnet18' / 'se_resnext50_32x4d'
batch_size = args.batch_size # 64/32

if mode == 'train':
    train_dataset = random_stack_dataset.RandomStackSegmentationDataset(
            config.paths['random_stack'], 'train', subset=args.train_subset)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

test_dataset = random_stack_dataset.RandomStackSegmentationDataset(
        config.paths['random_stack'], 'test', subset=args.test_subset)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = segmentation_models_pytorch.FPN(
        encoder_name = model_type,
        encoder_weights = args.encoder_weights,
        classes = 7 + args.include_confidence*2,
        activation = None).cuda()

optim = torch.optim.Adam(model.parameters(), lr=3e-4)

num_epochs = args.num_epochs

def test_epoch(epoch):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        correct_no_background = 0
        total_no_background = 0
        test_iterate = tqdm.tqdm(test_loader)
        for i, (images, targets) in enumerate(test_iterate):
            images = images.cuda()
            targets = targets.cuda()
            if args.include_confidence:
                logits_confidence = model(images)
                logits = logits_confidence[:,:-2]
                confidence = torch.softmax(logits_confidence[:,2:], dim=1)[:,1]
            else:
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
                    '''
                    mask_color = masks.color_floats_to_ints(
                            masks.index_to_mask_color(j))
                    '''
                    mask_color = masks.color_index_to_byte(j)
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
                    
                    if args.include_confidence:
                        confidence_bytes = confidence[j].cpu().numpy() * 255
                        confidence_bytes = confidence_bytes.astype(numpy.uint8)
                        confidence_image = Image.fromarray(confidence_bytes)
                        confidence_image.save(
                                './confidence_%i_%i.png'%(epoch, j))
        
        print('Accuracy:               %f'%(float(correct)/total))
        print('Accuracy No Background: %f'%(
                float(correct_no_background)/float(total_no_background)))

if mode == 'train':
    for epoch in range(1, num_epochs+1):
        print('Epoch: %i'%epoch)
        print('Train')
        model.train()
        train_iterate = tqdm.tqdm(train_loader)
        for images, targets in train_iterate:
            images = images.cuda()
            targets = targets.cuda()
            if args.include_confidence:
                logits_confidence = model(images)
                logits = logits_confidence[:,:-2]
                classification_loss = torch.nn.functional.cross_entropy(
                        logits, targets)
                
                predictions = torch.argmax(logits, dim=1).detach()
                confidence_target = (predictions == targets) & (targets != 0)
                confidence = logits_confidence[:,2:]
                confidence_loss = torch.nn.functional.cross_entropy(
                        confidence, confidence_target.long())
                
                loss = classification_loss + confidence_loss
            else:
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
    checkpoint = ('./segmentation_checkpoint_%04i.pt'%args.num_epochs)
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    test_epoch(10)
