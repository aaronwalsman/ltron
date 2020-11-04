#!/usr/bin/env python
import time
import random
import math
import argparse

import torch
from torchvision.transforms.functional import to_tensor

import numpy

import PIL.Image as Image

import tqdm

import segmentation_models_pytorch

import renderpy.masks as masks

import brick_gym.config as config
from brick_gym.dataset.data_paths import data_paths
import brick_gym.dataset.ldraw_environment as ldraw_environment
import brick_gym.viewpoint.azimuth_elevation as azimuth_elevation
import brick_gym.random_stack.dataset as random_stack_dataset

# Read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
        '--encoder', type=str, default='se_resnext50_32x4d')#default='resnet34')
parser.add_argument(
        '--batch-size', type=int, default=32)
parser.add_argument(
        '--test', action='store_true')
parser.add_argument(
        '--num-epochs', type=int, default=300)
parser.add_argument(
        '--train-subset', type=int, default=None)
parser.add_argument(
        '--test-subset', type=int, default=None)
parser.add_argument(
        '--episodes-per-train-epoch', type=int, default=256)
parser.add_argument(
        '--episodes-per-test-epoch', type=int, default=64)
parser.add_argument(
        '--num-mini-epochs', type=int, default=3)
parser.add_argument(
        '--encoder-weights', type=str, default='imagenet')
parser.add_argument(
        '--class-weight', type=float, default=0.8)
parser.add_argument(
        '--confidence-weight', type=float, default=0.2)
parser.add_argument(
        '--image-size', type=str, default='256x256')
parser.add_argument(
        '--checkpoint-frequency', type=int, default=10)
parser.add_argument(
        '--test-frequency', type=int, default=1)
args = parser.parse_args()

# Build the data generators
if not args.test:
    train_paths = data_paths(config.paths['random_stack'], 'train_mpd')

test_paths = data_paths(config.paths['random_stack'], 'test_mpd')

width, height = args.image_size.split('x')
width = int(width)
height = int(height)
viewpoint_control = azimuth_elevation.FixedAzimuthalViewpoint(
        azimuth = math.radians(30), elevation = -math.radians(45))
environment = ldraw_environment.LDrawEnvironment(
        viewpoint_control,
        width = width,
        height = height)

# Build the model
model = segmentation_models_pytorch.FPN(
        encoder_name = args.encoder,
        encoder_weights = args.encoder_weights,
        classes = 7 + 2,
        activation = None).cuda()

# Build the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

#brick_weights = torch.FloatTensor([0.02, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4]).cuda()
brick_weights = torch.FloatTensor([0.02, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4]).cuda()
confidence_weights = torch.FloatTensor([1, 0.01]).cuda()

def rollout(
        epoch,
        model_paths,
        num_episodes,
        steps_per_episode=8,
        ground_truth_foreground=True,
        dump_images=False,
        mark_selection=False):
    model.eval()
    with torch.no_grad():
        images = torch.zeros(
                num_episodes, steps_per_episode+1, 3, height, width)
        targets = torch.zeros(
                num_episodes, steps_per_episode+1, height, width,
                dtype=torch.long)
        logits = torch.zeros(
                num_episodes, steps_per_episode, 7+2, height, width)
        actions = torch.zeros(
                num_episodes, steps_per_episode, 2, dtype=torch.long)
        valid_entries = []
        for episode in tqdm.tqdm(range(num_episodes)):
            model_path = random.choice(model_paths)
            environment.load_path(model_path)
            image_numpy = environment.reset()
            if dump_images:
                Image.fromarray(image_numpy).save(
                        './image_%i_%i_0.png'%(epoch, episode))
            image = to_tensor(image_numpy)
            images[episode, 0] = image
            mask = environment.observe('mask')
            if dump_images:
                Image.fromarray(mask).save(
                        './mask_%i_%i_0.png'%(epoch, episode))
            target = masks.color_byte_to_index(mask)
            targets[episode, 0] = torch.LongTensor(target)
            for step in range(steps_per_episode):
                # cudafy
                image = image.cuda().unsqueeze(0)
                #mask = mask.cuda().unsqueeze(0)
                
                # forward pass
                step_logits = model(image)
                logits[episode, step] = step_logits
                
                # estimated segmentation
                segmentation_logits = step_logits[:,:-2]
                segmentation_indices = torch.argmax(segmentation_logits, dim=1)
                if dump_images:
                    prediction_mask = masks.color_index_to_byte(
                            segmentation_indices[0].cpu().numpy())
                    Image.fromarray(prediction_mask).save(
                            './pred_%i_%i_%i.png'%(epoch, episode, step))
                
                # estimated confidence
                confidence_logits = step_logits[:,-2:]
                confidence = torch.softmax(confidence_logits, dim=1)[:,1]
                if dump_images:
                    conf = (confidence[0] * 255).type(torch.uint8).cpu().numpy()
                    Image.fromarray(conf).save(
                            './conf_%i_%i_%i.png'%(epoch, episode, step))
                    
                    conf_target = (
                            (segmentation_indices[0].cpu().numpy() == target))
                    Image.fromarray((conf_target*255).astype(numpy.uint8)).save(
                            './conf_target_%i_%i_%i.png'%(epoch, episode, step))
                
                # select pixel-level action
                h, w = confidence.shape[-2:]
                if ground_truth_foreground:
                    foreground = torch.LongTensor(target != 0).cuda()
                else:
                    foreground = segmentation_indices != 0
                stop = int(torch.sum(foreground).cpu()) == 0
                if stop:
                    break
                foreground_confidence = confidence * foreground
                location = int(torch.argmax(foreground_confidence).cpu())
                y, x = numpy.unravel_index(location, (height, width))
                environment.hide_brick_at_pixel(x, y)
                
                if dump_images:
                    marked_numpy = image_numpy.copy()
                    marked_numpy[y-5:y+5,x] = (255,0,0)
                    marked_numpy[y,x-5:x+5] = (255,0,0)
                    Image.fromarray(marked_numpy).save(
                            './marked_%i_%i_%i.png'%(epoch, episode, step))
                
                # update data
                image_numpy = environment.observe('color')
                if dump_images:
                    Image.fromarray(image_numpy).save(
                            './image_%i_%i_%i.png'%(epoch, episode, step+1))
                image = to_tensor(image_numpy)
                images[episode, step+1] = image
                mask = environment.observe('mask')
                if dump_images:
                    Image.fromarray(mask).save(
                            './mask_%i_%i_%i.png'%(epoch, episode, step+1))
                target = masks.color_byte_to_index(mask)
                targets[episode, step+1] = torch.LongTensor(target)
                actions[episode, step, 0] = x
                actions[episode, step, 1] = y
                valid_entries.append((episode, step))
        
        return images, targets, logits, actions, valid_entries

# Train an epoch
def train_epoch(epoch):
    print('Train Epoch: %i'%epoch)
    
    # generate some data with the latest model
    print('Generating Data')
    images, targets, _, actions, valid_entries = rollout(
            epoch, train_paths, args.episodes_per_train_epoch)
    
    # train on the newly generated data
    for mini_epoch in range(1, args.num_mini_epochs+1):
        print('Train Mini Epoch: %i'%mini_epoch)
        random.shuffle(valid_entries)
        train_iterate = tqdm.tqdm(range(0, len(valid_entries), args.batch_size))
        for i in train_iterate:
            entries = valid_entries[i:i+args.batch_size]
            episodes, steps = zip(*entries)
            batch_images = images[episodes, steps]
            batch_targets = targets[episodes, steps]
            
            # cudafy data
            batch_images = batch_images.cuda()
            batch_targets = batch_targets.cuda()
            
            # forward
            logits = model(batch_images)
            
            # brick class loss
            class_logits = logits[:,:-2]
            class_loss = torch.nn.functional.cross_entropy(
                    class_logits, batch_targets, weight=brick_weights)
            
            # confidence loss
            confidence_logits = logits[:,-2:]
            
            predictions = torch.argmax(logits, dim=1).detach()
            confidence_target = (predictions == batch_targets)
            
            confidence_loss = torch.nn.functional.cross_entropy(
                    confidence_logits,
                    confidence_target.long(),
                    weight=confidence_weights)
            confidence_loss = torch.mean(confidence_loss)
            
            # combine loss
            loss = (class_loss * args.class_weight +
                    confidence_loss * args.confidence_weight)
            
            # backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_iterate.set_description(
                    'Seg: %.04f Conf: %.04f'%(
                        float(class_loss), float(confidence_loss)))

# Test an epoch
def test_epoch(epoch, dump_images=False, mark_selection=False):
    print('Test Epoch: %i'%epoch)
    print('Generating Data')
    images, targets, logits, actions, valid_entries = rollout(
            epoch,
            test_paths,
            args.episodes_per_test_epoch,
            ground_truth_foreground = False,
            dump_images = dump_images,
            mark_selection = mark_selection)
    episodes, steps = zip(*valid_entries)
    images = images[episodes, steps]
    valid_targets = targets[episodes, steps]
    valid_logits = logits[episodes, steps]
    #batch_size, steps, c, h, w = images.shape
    #images = images.view(batch_size * steps, c, h, w)
    #targets = targets.view(batch_size * steps, c, h, w)
    #logits = logits.view(batch_size * steps, c, h, w)
    
    seg_correct = 0
    seg_total = 0
    seg_correct_foreground = 0
    seg_total_foreground = 0
    
    class_logits = valid_logits[:,:-2]
    confidence_logits = valid_logits[:,-2:]
    
    class_predictions = torch.argmax(class_logits, dim=1)
    seg_correct = class_predictions == valid_targets
    seg_total = valid_targets.numel()
    
    print('Accuracy: %f'%(float(torch.sum(seg_correct)) / seg_total))
    
    foreground = (valid_targets != 0).long()
    seg_correct_foreground = seg_correct * foreground
    print('Foreground Accuracy: %f'%(
            float(torch.sum(seg_correct_foreground)) /
            float(torch.sum(foreground))))
    print('Foreground Ratio: %f'%(
            float(torch.sum(foreground))/torch.numel(foreground)))
    
    step_correct = 0
    iterate = tqdm.tqdm(valid_entries)
    for i, (episode, step) in enumerate(iterate):
        x, y = actions[episode, step]
        pixel_logits = logits[episode, step, :, y, x]
        prediction = torch.argmax(pixel_logits)
        target = targets[episode, step, y, x]
        step_correct += int(prediction == target)
        iterate.set_description('Acc: %f'%(step_correct/(i+1)))
    
    print('Step Accuracy: %f'%(step_correct/len(valid_entries)))

if args.test:
    checkpoint = './checkpoint_%04i.pt'%args.num_epochs
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    test_epoch(args.num_epochs, dump_images=True, mark_selection=True)

else:
    for epoch in range(1, args.num_epochs+1):
        t0 = time.time()
        # train
        train_epoch(epoch)
        
        # save
        if epoch % args.checkpoint_frequency == 0:
            checkpoint_path = './checkpoint_%04i.pt'%epoch
            print('Saving Checkpoint to: %s'%checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path)
        
        # test
        if epoch % args.test_frequency == 0:
            test_epoch(epoch)
        print('Elapsed: %.04f'%(time.time() - t0))
