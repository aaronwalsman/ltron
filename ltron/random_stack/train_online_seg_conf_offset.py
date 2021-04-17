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

max_bricks_per_scene = 8

# Read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
        '--encoder', type=str, default='se_resnext50_32x4d')#default='resnet34')
parser.add_argument(
        '--batch-size', type=int, default=8)
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
        '--offset-weight', type=float, default=5000.0)
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
downsample = 1./8
downsample_width = int(width * downsample)
downsample_height = int(height * downsample)
downsample_stride_x = width // downsample_width
downsample_stride_y = height // downsample_height
viewpoint_control = azimuth_elevation.FixedAzimuthalViewpoint(
        azimuth = math.radians(30), elevation = -math.radians(45))

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
        activation = None)
model = offset2d.FCNOffset2D(
        fcn,
        feature_channels,
        3,
        7 + 2,
        downsample = downsample).cuda()

# Build the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#brick_weights = torch.FloatTensor([0.02, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4]).cuda()
brick_weights = torch.FloatTensor([0.01, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4]).cuda()
existence_weights = torch.FloatTensor([0.1, 1.0]).cuda()
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
    accumulator = torch.nn.functional.avg_pool2d(
            accumulator,
            (downsample_stride_y, downsample_stride_x),
            (downsample_stride_y, downsample_stride_x))
    non_empty = accumulator[0] != 1.0
    result = (torch.argmax(accumulator[1:], dim=0) + 1) * non_empty
    return result.cpu().numpy()

def rollout(
        epoch,
        model_paths,
        num_episodes,
        steps_per_episode=8,
        ground_truth_foreground=True):
    
    model.eval()
    with torch.no_grad():
        images = torch.zeros(
                num_episodes, steps_per_episode+1, 3, height, width)
        offset_targets = torch.zeros(
                num_episodes,
                steps_per_episode+1,
                2,
                height,
                width)
        centers = torch.zeros(
                num_episodes, steps_per_episode+1, max_bricks_per_scene, 2)
        center_indices = torch.zeros(
                num_episodes, steps_per_episode+1, max_bricks_per_scene)
        logits = torch.zeros(
                num_episodes, steps_per_episode, 7+2,
                downsample_height, downsample_width)
        actions = torch.zeros(
                num_episodes, steps_per_episode, 3, dtype=torch.long)
        valid_entries = []
        model_paths = random.sample(model_paths, num_episodes)
        for ep_start in tqdm.tqdm(range(0, num_episodes, args.num_processes)):
            ep_end = ep_start + args.num_processes
            batch_model_paths = model_paths[ep_start:ep_end]
            multi_environment.load_paths(batch_model_paths)
            instance_brick_types = multi_environment.get_instance_brick_types()
            instance_class_lookup = [
                    get_instance_class_lookup(brick_types)
                    for brick_types in instance_brick_types]
            observations = multi_environment.observe(('color', 'instance_mask'))
            batch_images, batch_masks = zip(*observations)
            
            instance_ids = [
                    masks.color_byte_to_index(mask)
                    for mask in batch_masks]
            batch_offset_targets = [
                    offset2d.offset_targets(
                        torch.LongTensor(ids), range(1, max_bricks_per_scene+1))
                    for ids in instance_ids]
            valid_entries.extend(
                    [(ep_start+b,0) for b in range(len(batch_model_paths))])
            
            if args.dump_images:
                for i, (image, mask) in enumerate(observations):
                    episode = ep_start + i
                    Image.fromarray(image).save(
                            './image_%i_%i_0.png'%(epoch, episode))
                    Image.fromarray(mask).save(
                            './instance_mask_%i_%i_0.png'%(epoch, episode))
                    step_offsets, step_centers, step_center_indices = (
                            batch_offset_targets[i])
                    step_offsets = step_offsets.permute(1,2,0) * 0.5 + 0.5
                    offset_image = numpy.zeros(
                            (height, width, 3), dtype=numpy.uint8)
                    offset_image[:,:,:2] = (
                            step_offsets * 255).type(torch.uint8).numpy()
                    offset_image = offset_image * (mask != 0)
                    for j  in range(step_centers.shape[0]):
                        if step_center_indices[j]:
                            c = step_centers[j]
                            y = int(torch.floor(c[0] * height))
                            x = int(torch.floor(c[1] * height))
                            offset_image[y-5:y+5,x] = (0,0,255)
                            offset_image[y,x-5:x+5] = (0,0,255)
                    Image.fromarray(offset_image).save(
                            './offset_target_%i_%i_0.png'%(epoch, episode))
            
            # store images
            batch_images = torch.stack(
                    [to_tensor(image) for image in batch_images], dim=0)
            images[ep_start:ep_end, 0] = batch_images
            
            # store offsets, centers and center indices
            all_offsets, all_centers, all_indices = zip(*batch_offset_targets)
            all_indices = [
                    torch.LongTensor(lookup)[indices]
                    for indices, lookup
                    in zip(all_indices, instance_class_lookup)]
            offset_targets[ep_start:ep_end, 0] = torch.stack(all_offsets, dim=0)
            centers[ep_start:ep_end, 0] = torch.stack(all_centers, dim=0)
            center_indices[ep_start:ep_end, 0] = torch.stack(all_indices, dim=0)
            
            for step in range(steps_per_episode):
                # cudafy
                batch_images = batch_images.cuda()
                
                # forward pass
                step_logits, offset, attention, destination = model(
                        batch_images)
                logits[ep_start:ep_end, step] = step_logits
                
                if args.dump_images:
                    for i in range(offset.shape[0]):
                        step_offsets = offset[i]
                        step_offsets = step_offsets.permute(1,2,0) * 5.5 + 0.5
                        offset_image = numpy.zeros(
                                (height, width, 3), dtype=numpy.uint8)
                        offset_image[:,:,:2] = (
                                step_offsets*255).type(
                                torch.uint8).cpu().numpy()
                        Image.fromarray(offset_image).save(
                                './offset_pred_%i_%i_%i.png'%(
                                    epoch, ep_start+i, step))
                
                # estimated class labels
                segmentation_logits = step_logits[:,:-2]
                segmentation_indices = torch.argmax(segmentation_logits, dim=1)
                if args.dump_images:
                    for i in range(segmentation_indices.shape[0]):
                        episode = ep_start + i
                        prediction_mask = masks.color_index_to_byte(
                                segmentation_indices[i].cpu().numpy())
                        Image.fromarray(prediction_mask).save(
                                './pred_%i_%i_%i.png'%(epoch, episode, step))
                
                    # ground truth class_labels
                    h, w = segmentation_logits.shape[-2:]
                    for b in range(segmentation_indices.shape[0]):
                        class_targets = numpy.zeros((h,w), dtype=numpy.long)
                        for j in range(max_bricks_per_scene):
                            y, x = all_centers[b][j]
                            y = int(y * h)
                            x = int(x * w)
                            brick_id = all_indices[b][j]
                            class_targets[y, x] = brick_id
                        class_target_mask = masks.color_index_to_byte(
                                class_targets)
                        Image.fromarray(class_target_mask).save(
                                './class_target_%i_%i_%i.png'%(
                                    epoch, episode, step))
                        
                
                # estimated confidence
                confidence_logits = step_logits[:,-2:]
                confidence = torch.softmax(confidence_logits, dim=1)[:,1]
                if args.dump_images:
                    for i in range(confidence.shape[0]):
                        episode = ep_start + i
                        conf = (confidence[i] * 255).type(torch.uint8).cpu()
                        Image.fromarray(conf.numpy()).save(
                                './conf_%i_%i_%i.png'%(epoch, episode, step))
                
                # select brick-level action
                bs, h, w = confidence.shape
                #gt_foreground = torch.zeros(bs, h, w)
                ground_truth_bricks = []
                brick_confidence = []
                for b, (_, center, indices) in enumerate(batch_offset_targets):
                    batch_gt_bricks = []
                    batch_confidence = []
                    for j in range(center.shape[0]):
                        brick_index = int(indices[j])
                        if brick_index:
                            y = int(center[j,0] * h)
                            x = int(center[j,1] * w)
                            #gt_foreground[b,y,x] = 1
                            batch_gt_bricks.append((y,x,brick_index))
                            batch_confidence.append(float(confidence[b,y,x]))
                    ground_truth_bricks.append(batch_gt_bricks)
                    brick_confidence.append(torch.FloatTensor(batch_confidence))
                stop = torch.zeros(bs, dtype=torch.bool)
                
                hide_bricks = []
                batch_actions = torch.zeros(bs, 3, dtype=torch.long)
                if ground_truth_foreground:
                    for b, (bricks, confs) in enumerate(zip(
                            ground_truth_bricks, brick_confidence)):
                        if len(bricks):
                            max_conf_id = torch.argmax(confs)
                            y, x, brick_id = bricks[max_conf_id]
                            hide_bricks.append(brick_id)
                            batch_actions[b] = torch.LongTensor(
                                    [y, x, brick_id])
                        else:
                            stop[b] = 1
                            hide_bricks.append(0)
                else:
                    foreground = segmentation_indices != 0
                    foreground_confidence = confidence * foreground
                    foreground_confidence = foreground_confidence.view(
                            foreground_confidence.shape[0], -1)
                    locations = torch.argmax(
                            foreground_confidence, dim=-1).cpu()
                    y, x = numpy.unravel_index(
                            locations, (downsample_height, downsample_width))
                    for b in range(y.shape[0]):
                        yy = int(y[b])
                        xx = int(x[b])
                        brick_id = int(segmentation_indices[b,yy,xx])
                        if (yy, xx, brick_id) in ground_truth_bricks[b]:
                            hide_bricks.append(brick_id)
                        else:
                            stop[b] = 1
                            hide_bricks.append(0)
                        batch_actions[b][0] = yy
                        batch_actions[b][1] = xx
                        batch_actions[b][2] = brick_id
                '''
                if ground_truth_foreground:
                    foreground = gt_foreground
                else:
                    foreground = segmentation_indices != 0
                '''
                '''
                h, w = confidence.shape[-2:]
                if ground_truth_foreground:
                    foreground = (batch_class_targets != 0).long().cuda()
                else:
                    foreground = segmentation_indices != 0
                '''
                #stop = torch.sum(foreground, dim=(1,2)).cpu() == 0
                '''
                foreground_confidence = confidence * foreground
                foreground_confidence = foreground_confidence.view(
                        foreground_confidence.shape[0], -1)
                locations = torch.argmax(foreground_confidence, dim=-1).cpu()
                y, x = numpy.unravel_index(
                        locations, (downsample_height, downsample_width))
                '''
                
                #hide_ids = [ids[y[i],x[i]]
                #        for i, ids in enumerate(instance_ids)]
                multi_environment.hide_bricks(hide_bricks)
                
                '''
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
                '''
                
                # update data
                observations = multi_environment.observe(
                        ('color', 'instance_mask'))
                batch_images, batch_masks = zip(*observations)
                instance_ids = [
                        masks.color_byte_to_index(mask)
                        for mask in batch_masks]
                batch_offset_targets = [
                        offset2d.offset_targets(
                            torch.LongTensor(ids),
                            range(1, max_bricks_per_scene+1))
                        for ids in instance_ids]
                
                if args.dump_images:
                    for i, (image, mask) in enumerate(observations):
                        episode = ep_start + i
                        Image.fromarray(image).save(
                                './image_%i_%i_%i.png'%(
                                    epoch, episode, step+1))
                        Image.fromarray(mask).save(
                                './instance_mask_%i_%i_%i.png'%(
                                    epoch, episode, step+1))
                        step_offsets, step_centers, step_center_indices = (
                                batch_offset_targets[i])
                        step_offsets = step_offsets.permute(1,2,0) * 0.5 + 0.5
                        offset_image = numpy.zeros(
                                (height, width, 3), dtype=numpy.uint8)
                        offset_image[:,:,:2] = (
                                step_offsets * 255).type(torch.uint8).numpy()
                        offset_image = offset_image * (mask != 0)
                        for j  in range(step_centers.shape[0]):
                            if step_center_indices[j]:
                                c = step_centers[j]
                                y = int(torch.floor(c[0] * height))
                                x = int(torch.floor(c[1] * height))
                                offset_image[y-5:y+5,x] = (0,0,255)
                                offset_image[y,x-5:x+5] = (0,0,255)
                        Image.fromarray(offset_image).save(
                                './offset_target_%i_%i_%i.png'%(
                                    epoch, episode, step+1))
                
                # store images
                batch_images = torch.stack(
                        [to_tensor(image) for image in batch_images], dim=0)
                images[ep_start:ep_end, step+1] = batch_images
                
                # store offsets, centers and center indices
                all_offsets, all_centers, all_indices = zip(
                        *batch_offset_targets)
                all_indices = [
                        torch.LongTensor(lookup)[indices]
                        for indices, lookup
                        in zip(all_indices, instance_class_lookup)]
                offset_targets[ep_start:ep_end, step+1] = torch.stack(
                        all_offsets, dim=0)
                centers[ep_start:ep_end, step+1] = torch.stack(
                        all_centers, dim=0)
                center_indices[ep_start:ep_end, step+1] = torch.stack(
                        all_indices, dim=0)
                
                '''
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
                '''
                actions[ep_start:ep_end, step] = batch_actions
                valid_entries.extend([
                        (ep_start + b, step+1)
                        for b in range(batch_images.shape[0])
                        if not stop[j]])
        
        return (images,
                offset_targets,
                centers,
                center_indices,
                logits,
                actions,
                valid_entries)

# Train an epoch
def train_epoch(epoch):
    print('Train Epoch: %i'%epoch)
    
    # generate some data with the latest model
    print('Generating Data')
    (images,
     offset_targets,
     centers,
     center_indices,
     _,
     actions,
     valid_entries) = rollout(
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
            batch_offset_targets = offset_targets[episodes, steps]
            batch_centers = centers[episodes, steps]
            batch_center_indices = center_indices[episodes, steps]
            
            # cudafy data
            batch_images = batch_images.cuda()
            batch_offset_targets = batch_offset_targets.cuda()
            
            # forward
            logits, offsets, attention, destination = model(batch_images)
            
            # brick class loss
            class_logits = logits[:,:-2]
            bs, _, h, w = class_logits.shape
            batch_class_targets = torch.zeros(bs, h, w, dtype=torch.long)
            for b in range(bs):
                for j in range(max_bricks_per_scene):
                    y, x = batch_centers[b,j]
                    y = int(y * h)
                    x = int(x * w)
                    brick_id = batch_center_indices[b,j]
                    batch_class_targets[b, y, x] = brick_id
            batch_class_targets = batch_class_targets.cuda()
            class_loss = torch.nn.functional.cross_entropy(
                    class_logits, batch_class_targets, weight=brick_weights)
            
            # offset loss
            offset_loss = torch.nn.functional.mse_loss(
                    offsets, batch_offset_targets)
            
            # confidence loss
            confidence_logits = logits[:,-2:]
            
            predictions = torch.argmax(class_logits, dim=1).detach()
            confidence_target = (
                    (predictions == batch_class_targets) * 
                    (batch_class_targets != 0))
            
            confidence_loss = torch.nn.functional.cross_entropy(
                    confidence_logits,
                    confidence_target.long(),
                    weight=confidence_weights)
            #confidence_loss = torch.mean(confidence_loss)
            
            # combine loss
            loss = (class_loss * args.class_weight +
                    offset_loss * args.offset_weight +
                    confidence_loss * args.confidence_weight)
            
            # backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_iterate.set_description(
                    'Seg: %.04f Off: %.04f Conf: %.04f'%(
                        float(class_loss),
                        float(offset_loss),
                        float(confidence_loss)))

# Test an epoch
def test_epoch(epoch):
    print('Test Epoch: %i'%epoch)
    print('Generating Data')
    (images,
     offset_targets,
     centers,
     center_indices,
     logits,
     actions,
     valid_entries) = rollout(
            epoch,
            test_paths,
            args.episodes_per_test_epoch,
            ground_truth_foreground = False)
    if not len(valid_entries):
        print('No good data yet')
        return
    
    episodes, steps = zip(*valid_entries)
    images = images[episodes, steps]
    valid_offset_targets = offset_targets[episodes, steps]
    valid_logits = logits[episodes, steps]
    valid_centers = centers[episodes, steps]
    valid_center_indices = center_indices[episodes, steps]
    #batch_size, steps, c, h, w = images.shape
    #images = images.view(batch_size * steps, c, h, w)
    #targets = targets.view(batch_size * steps, c, h, w)
    #logits = logits.view(batch_size * steps, c, h, w)
    
    bs, _, h, w = valid_logits.shape
    valid_class_targets = torch.zeros(bs, h, w)
    for b in range(bs):
        for j in range(max_bricks_per_scene):
            y = valid_centers[b,j,0]
            x = valid_centers[b,j,1]
            y = int(y*h)
            x = int(x*w)
            class_id = valid_center_indices[b,j]
            valid_class_targets[b,y,x] = class_id
    
    seg_correct = 0
    seg_total = 0
    seg_correct_foreground = 0
    seg_total_foreground = 0
    
    class_logits = valid_logits[:,:-2]
    confidence_logits = valid_logits[:,-2:]
    
    class_predictions = torch.argmax(class_logits, dim=1)
    seg_correct = class_predictions == valid_class_targets
    seg_total = valid_class_targets.numel()
    
    print('Accuracy: %f'%(float(torch.sum(seg_correct)) / seg_total))
    
    foreground = (valid_class_targets != 0).long()
    seg_correct_foreground = seg_correct * foreground
    print('Foreground Accuracy: %f'%(
            float(torch.sum(seg_correct_foreground)) /
            float(torch.sum(foreground))))
    print('Foreground Ratio: %f'%(
            float(torch.sum(foreground))/torch.numel(foreground)))
    
    step_correct = 0
    iterate = tqdm.tqdm(valid_entries)
    for i, (episode, step) in enumerate(iterate):
        correct_actions = []
        for j in range(max_bricks_per_scene):
            brick_class = center_indices[episode, step, j]
            if brick_class:
                y = int(centers[episode, step, j, 0] * h)
                x = int(centers[episode, step, j, 1] * w)
                correct_actions.append((y, x, brick_class))
        y, x, prediction = actions[episode, step]
        pixel_logits = logits[episode, step, :, y, x]
        #prediction = torch.argmax(pixel_logits)
        #target = class_targets[episode, step, y, x]
        #target = valid_class_targets[i, y, x]
        step_correct += (y, x, prediction) in correct_actions
        iterate.set_description('Acc: %f'%(step_correct/(i+1)))
    
    print('Step Accuracy: %f'%(step_correct/len(valid_entries)))

with multi_environment:
    if args.test:
        checkpoint = './checkpoint_%04i.pt'%args.num_epochs
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        test_epoch(args.num_epochs)

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
