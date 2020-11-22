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
import brick_gym.model.offset2d as offset2d
#from brick_gym.random_stack.HarDNet import hardnet

import renderpy.masks as masks

import brick_gym.config as config
from brick_gym.dataset.data_paths import data_paths
import brick_gym.dataset.ldraw_environment as ldraw_environment
import brick_gym.viewpoint.azimuth_elevation as azimuth_elevation
import brick_gym.random_stack.dataset as random_stack_dataset

mesh_indices = {
    '3005' : 1,
    '3004' : 2,
    '3003' : 3,
    '3002' : 4,
    '3001' : 5,
    '2456' : 6}

# Read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
        '--encoder', type=str, default='se_resnext50_32x4d')#default='resnet34')
parser.add_argument(
        '--batch-size', type=int, default=32)
parser.add_argument(
        '--lr', type=float, default=3e-4)
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
parser.add_argument(
        '--dump-images', action='store_true')
parser.add_argument(
        '--num-processes', type=int, default=16)
args = parser.parse_args()

# Build the data generators
if not args.test:
    train_paths = data_paths(config.paths['random_stack'], 'train_mpd')

test_paths = data_paths(config.paths['random_stack'], 'test_mpd')

width, height = args.image_size.split('x')
width = int(width)
height = int(height)
downsample_width = 32
downsample_height = 32
viewpoint_control = azimuth_elevation.FixedAzimuthalViewpoint(
        azimuth = math.radians(30), elevation = -math.radians(45))
'''
environment = ldraw_environment.LDrawEnvironment(
        viewpoint_control,
        width = width,
        height = height)
'''

multi_environment = ldraw_environment.MultiLDrawEnvironment(
        num_processes = args.num_processes,
        width = width,
        height = height,
        viewpoint_control = viewpoint_control)

# Build the model
feature_channels = 256
fcn = segmentation_models_pytorch.FPN(
        encoder_name = args.encoder,
        encoder_weights = args.encoder_weights,
        classes = feature_channels,
        activation = None).cuda()
model = offset2d.Offset2DSegmentationModel(
        fcn,
        feature_channels,
        7 + 2 + 2)

# Build the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#brick_weights = torch.FloatTensor([0.02, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4]).cuda()
brick_weights = torch.FloatTensor([0.02, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4]).cuda()
confidence_weights = torch.FloatTensor([1, 0.01]).cuda()

def get_instance_class_lookup(instance_brick_types):
    lookup = numpy.zeros(1 + len(instance_brick_types), dtype=numpy.long)
    for instance_id, mesh_name in instance_brick_types.items():
        lookup[instance_id] = mesh_indices[mesh_name]
    return lookup

def downsample_indices(indices, num_classes=9):
    indices = torch.LongTensor(indices).cuda().unsqueeze(0)
    accumulator = torch.zeros(num_classes, 256, 256).cuda()
    accumulator.scatter_(0, indices, 1)
    accumulator = torch.nn.functional.avg_pool2d(accumulator, 32, 32)
    non_empty = accumulator[0] != 1.0
    result = (torch.argmax(accumulator[1:], dim=0) + 1) * non_empty
    return result.cpu().numpy()

def rollout(
        epoch,
        model_paths,
        num_episodes,
        steps_per_episode=8,
        ground_truth_foreground=True,
        mark_selection=False):
    
    model.eval()
    with torch.no_grad():
        images = torch.zeros(
                num_episodes, steps_per_episode+1, 3, height, width)
        class_targets = torch.zeros(
                num_episodes,
                steps_per_episode+1,
                downsample_height,
                downsample_width,
                dtype=torch.long)
        logits = torch.zeros(
                num_episodes, steps_per_episode, 7+2,
                downsample_height, downsample_width)
        actions = torch.zeros(
                num_episodes, steps_per_episode, 2, dtype=torch.long)
        valid_entries = []
        #for episode in tqdm.tqdm(range(num_episodes)):
        model_paths = random.sample(model_paths, num_episodes)
        for ep_start in tqdm.tqdm(range(0, num_episodes, args.num_processes)):
            ep_end = ep_start + args.num_processes
            #model_path = random.choice(model_paths)
            #environment.load_path(model_path)
            batch_model_paths = model_paths[ep_start:ep_end]
            multi_environment.load_paths(batch_model_paths)
            instance_brick_types = multi_environment.get_instance_brick_types()
            instance_class_lookup = [
                    get_instance_class_lookup(brick_types)
                    for brick_types in instance_brick_types]
            #image_numpy = environment.reset()
            observations = multi_environment.observe(('color', 'instance_mask'))
            batch_images, batch_masks = zip(*observations)

            instance_ids = [
                    downsample_indices(masks.color_byte_to_index(mask))
                    for mask in batch_masks]
            batch_class_targets = [
                    class_lookup[ids]
                    for class_lookup, ids
                    in zip(instance_class_lookup, instance_ids)]
            
            if args.dump_images:
                for i, (image, mask) in enumerate(observations):
                    episode = ep_start + i
                    Image.fromarray(image).save(
                            './image_%i_%i_0.png'%(epoch, episode))
                    Image.fromarray(mask).save(
                            './instance_mask_%i_%i_0.png'%(epoch, episode))
                    class_mask = masks.color_index_to_byte(
                            batch_class_targets[i])
                    Image.fromarray(class_mask).save(
                            './class_mask_%i_%i_0.png'%(epoch, episode))
            
            batch_images = torch.stack(
                    [to_tensor(image) for image in batch_images], dim=0)
            images[ep_start:ep_end, 0] = batch_images
            #mask = environment.observe('mask')
            #if args.dump_images:
            #    Image.fromarray(mask).save(
            #            './mask_%i_%i_0.png'%(epoch, episode))
            #target = masks.color_byte_to_index(mask)
            #targets[episode, 0] = torch.LongTensor(target)
            #numpy_masks = [masks.color_byte_to_index(mask)
            #        for _, mask in numpy_images]
            batch_class_targets = torch.stack([torch.LongTensor(target)
                    for target in batch_class_targets], dim=0)
            class_targets[ep_start:ep_end, 0] = batch_class_targets
            for step in range(steps_per_episode):
                # cudafy
                #image = image.cuda().unsqueeze(0)
                #mask = mask.cuda().unsqueeze(0)
                batch_images = batch_images.cuda()
                
                # forward pass
                step_logits = model(batch_images)
                logits[ep_start:ep_end, step] = step_logits
                
                # estimated segmentation
                segmentation_logits = step_logits[:,:-2]
                segmentation_indices = torch.argmax(segmentation_logits, dim=1)
                if args.dump_images:
                    for i in range(segmentation_indices.shape[0]):
                        episode = ep_start + i
                        prediction_mask = masks.color_index_to_byte(
                                segmentation_indices[i].cpu().numpy())
                        Image.fromarray(prediction_mask).save(
                                './pred_%i_%i_%i.png'%(epoch, episode, step))
                
                # estimated confidence
                confidence_logits = step_logits[:,-2:]
                confidence = torch.softmax(confidence_logits, dim=1)[:,1]
                if args.dump_images:
                    for i in range(confidence.shape[0]):
                        episode = ep_start + i
                        conf = (confidence[i] * 255).type(torch.uint8).cpu()
                        Image.fromarray(conf.numpy()).save(
                                './conf_%i_%i_%i.png'%(epoch, episode, step))
                        
                        seg = segmentation_indices[i].cpu()
                        class_target = batch_class_targets[i]
                        conf_target = seg == class_target
                        conf_target_image = (
                                conf_target.numpy() * 255).astype(numpy.uint8)
                        Image.fromarray(conf_target_image).save(
                                './conf_target_%i_%i_%i.png'%(
                                epoch, episode, step))
                
                # select pixel-level action
                h, w = confidence.shape[-2:]
                if ground_truth_foreground:
                    foreground = (batch_class_targets != 0).long().cuda()
                else:
                    foreground = segmentation_indices != 0
                stop = torch.sum(foreground, dim=(1,2)).cpu() == 0
                
                foreground_confidence = confidence * foreground
                foreground_confidence = foreground_confidence.view(
                        foreground_confidence.shape[0], -1)
                locations = torch.argmax(foreground_confidence, dim=-1).cpu()
                y, x = numpy.unravel_index(
                        locations, (downsample_height, downsample_width))
                #xy = list(zip(x, y))
                '''
                location = int(torch.argmax(foreground_confidence).cpu())
                y, x = numpy.unravel_index(location, (height, width))
                environment.hide_brick_at_pixel(x, y)
                '''
                hide_ids = [ids[y[i],x[i]]
                        for i, ids in enumerate(instance_ids)]
                multi_environment.hide_bricks(hide_ids)
                
                if args.dump_images:
                    for i in range(batch_images.shape[0]):
                        episode = ep_start + i
                        image_numpy = observations[i][0]
                        marked_numpy = image_numpy.copy()
                        u_x = x * FACTOR + FACTOR/2
                        u_y = y * FACTOR + FACTOR/2
                        marked_numpy[u_y[i]-5:u_y[i]+5,u_x[i]] = (255,0,0)
                        marked_numpy[u_y[i],u_x[i]-5:u_x[i]+5] = (255,0,0)
                        Image.fromarray(marked_numpy).save(
                                './marked_%i_%i_%i.png'%(epoch, episode, step))
                
                # update data
                '''
                observations = environment.observe(('color', 'instance_mask'))
                batch_images, batch_masks = zip(*observations)
                if args.dump_images:
                    Image.fromarray(image_numpy).save(
                            './image_%i_%i_%i.png'%(epoch, episode, step+1))
                image = to_tensor(image_numpy)
                images[episode, step+1] = image
                mask = environment.observe('mask')
                if args.dump_images:
                    Image.fromarray(mask).save(
                            './mask_%i_%i_%i.png'%(epoch, episode, step+1))
                target = masks.color_byte_to_index(mask)
                targets[episode, step+1] = torch.LongTensor(target)
                actions[episode, step, 0] = x
                actions[episode, step, 1] = y
                valid_entries.append((episode, step))
                '''
                observations = multi_environment.observe(
                        ('color', 'instance_mask'))
                batch_images, batch_masks = zip(*observations)
                instance_ids = [
                        downsample_indices(masks.color_byte_to_index(mask))
                        for mask in batch_masks]
                batch_class_targets = [
                        class_lookup[ids]
                        for class_lookup, ids
                        in zip(instance_class_lookup, instance_ids)]
                
                if args.dump_images:
                    for i, (image, mask) in enumerate(observations):
                        episode = ep_start + i
                        Image.fromarray(image).save(
                                './image_%i_%i_%i.png'%(
                                epoch, episode, step+1))
                        Image.fromarray(mask).save(
                                './instance_mask_%i_%i_%i.png'%(
                                epoch, episode, step+1))
                        class_mask = masks.color_index_to_byte(
                                batch_class_targets[i])
                        Image.fromarray(class_mask).save(
                                './class_mask_%i_%i_%i.png'%(
                                epoch, episode, step+1))
                
                batch_images = torch.stack(
                        [to_tensor(image) for image in batch_images], dim=0)
                images[ep_start:ep_end, step+1] = batch_images
                batch_class_targets = torch.stack([torch.LongTensor(target)
                        for target in batch_class_targets], dim=0)
                class_targets[ep_start:ep_end, step+1] = batch_class_targets
                actions[ep_start:ep_end, step, 0] = torch.LongTensor(x)
                actions[ep_start:ep_end, step, 1] = torch.LongTensor(y)
                valid_entries.extend([
                        (ep_start + i, step)
                        for i in range(batch_images.shape[0])
                        if not stop[i]])
        
        return images, class_targets, logits, actions, valid_entries

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
            
            predictions = torch.argmax(class_logits, dim=1).detach()
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
def test_epoch(epoch, mark_selection=False):
    print('Test Epoch: %i'%epoch)
    print('Generating Data')
    images, targets, logits, actions, valid_entries = rollout(
            epoch,
            test_paths,
            args.episodes_per_test_epoch,
            ground_truth_foreground = False,
            mark_selection = mark_selection)
    if not len(valid_entries):
        print('No good data yet')
        return
    
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

with multi_environment:
    if args.test:
        checkpoint = './checkpoint_%04i.pt'%args.num_epochs
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        test_epoch(args.num_epochs, mark_selection=True)

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
