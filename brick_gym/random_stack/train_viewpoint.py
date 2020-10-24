#!/usr/bin/env python
import argparse

import torch
from torchvision.transforms.functional import to_tensor

import numpy

import PIL.Image as Image

import tqdm

import gym

import segmentation_models_pytorch

import renderpy.masks as masks

import brick_gym.config as config
import brick_gym.mpd_sequence as mpd_sequence
import brick_gym.viewpoint_control as viewpoint_control

parser = argparse.ArgumentParser()
parser.add_argument(
        '--test', action='store_true')
parser.add_argumnet(
        '--segmentation-model', type=str, default='resnet18')
parser.add_argument(
        '--segmentation-checkpoint', type=str)
parser.add_argument(
        '--num-epochs', type=int, default=10)
parser.add_argument(
        '--batches-per-epoch', type=int, default=5000)
parser.add_argument(
        '--batch-size', type=int, default=64)

args = parser.parse_args()

segmentation_model = segmentation_models_pytorch.FPN(
        encoder_name = 'se_resnext50_32x4d',
        encoder_weights = 'imagenet',
        classes = 7,
        activation = None).cuda()
checkpoint = (
        '../../runs/segmentation_train_001/segmentation_checkpoint_0010.pt')
state_dict = torch.load(checkpoint)
segmentation_model.load_state_dict(state_dict)

if not args.test:
    train_view_control = viewpoint_control.AzimuthElevationViewpointControl(
            reset_mode = 'random',
            elevation_range = [-1.0, 1.0])
    train_mpd_sequence = mpd_sequence.MPDSequence(
            train_view_control,
            directory = config.paths['random_stack'],
            split = 'train')

test_view_control = viewpoint_control.AzimuthElevationViewpointControl(
        elevation_range = [-1.0, 1.0])
test_mpd_sequence = mpd_sequence.MPDSequence(
        test_view_control,
        directory = config.paths['random_stack'],
        split = 'test')

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

viewpoint_model = ViewpointModel(segmentation_model).cuda()
if  args.test:
    weights = torch.load('viewpoint_checkpoint_0004.pt')
    viewpoint_model.load_state_dict(weights)

optimizer = torch.optim.Adam(viewpoint_model.parameters(), lr=3e-4)

def test_epoch(epoch):
    print('Testing Epoch: %i'%epoch)
    test_mpd_sequence.reset_state()
    action = None
    state = test_mpd_sequence.get_state()
    visited_states = set()
    accuracies = []
    with torch.no_grad():
        while action != 0 and state not in visited_states:
            image, mask = test_mpd_sequence.observe()
            x = to_tensor(image).unsqueeze(0).cuda()
            mask_logits = segmentation_model(x)
            mask_prediction = torch.argmax(mask_logits, dim=0)[0].cpu().numpy()
            y = masks.color_byte_to_index(mask)
            print(y.shape)
            print(mask_prediction.shape)
            correct = y == mask_prediction
            accuracy = numpy.sum(correct) / correct.size
            print(accuracy)
            accuracies.append(accuracy)
            
            logits = viewpoint_model(x)
            action = torch.argmax(logits).cpu()
            visited_states.add(state)
            test_mpd_sequence.viewpoint_control.step(action)
            state = test_mpd_sequence.get_state()
            

def train_batch():
    # pick a new scene
    train_mpd_sequence.reset_state()
    
    # render all the color images and target masks
    batch_images = []
    batch_masks = []
    for i in range(args.batch_size):
        # reset to a random viewpoint
        train_mpd_sequence.viewpoint_control.reset()
        viewpoint_state = train_mpd_sequence.viewpoint_control.get_state()
        image, mask = train_mpd_sequence.observe()
        '''
        ######
        Image.fromarray(image).save('tmp_image_%i_0.png'%i)
        Image.fromarray(mask).save('tmp_mask_%i_0.png'%i)
        ######
        '''
        batch_images.append(image)
        batch_masks.append(mask)
        for action in range(1,5):
            train_mpd_sequence.viewpoint_control.set_state(viewpoint_state)
            train_mpd_sequence.viewpoint_control.step(action)
            image, mask = train_mpd_sequence.observe()
            
            '''
            ########
            Image.fromarray(image).save('tmp_image_%i_%i.png'%(i, action))
            Image.fromarray(mask).save('tmp_mask_%i_%i.png'%(i, action))
            ########
            '''
            
            batch_images.append(image)
            batch_masks.append(mask)
    
    # get the predicted masks
    with torch.no_grad():
        costs = []
        for i in range(5):
            x = batch_images[i*args.batch_size:(i+1)*args.batch_size]
            x = [to_tensor(xx) for xx in x]
            x = torch.stack(x).cuda()
            logits = segmentation_model(x)
            
            '''
            ############
            prediction = torch.argmax(logits, dim=1)
            for j in range(batch_size):
                mask = masks.color_index_to_byte(prediction[j].cpu().numpy())
                idx = i * 16 + j
                view = idx // 5
                action = idx % 5
                Image.fromarray(mask).save(
                        'tmp_prediction_%i_%i.png'%(view, action))
            ############
            '''
            
            y = batch_masks[i*args.batch_size:(i+1)*args.batch_size]
            y = [torch.LongTensor(masks.color_byte_to_index(yy)) for yy in y]
            y = torch.stack(y).cuda()
            cost = torch.nn.functional.cross_entropy(
                    logits, y, reduction='none')
            cost = torch.mean(cost, dim=(1,2))
            costs.append(cost)
        costs = torch.cat(costs).view(args.batch_size, 5)
        #print(costs)
    
    # run the images through the action-space model
    x = batch_images[::5]
    x = [to_tensor(xx) for xx in x]
    x = torch.stack(x).cuda()
    logits = viewpoint_model(x)
    
    y = torch.argmin(costs, dim=1)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

if not args.test:
    for epoch in range(1, args.num_epochs+1):
        print('Epoch: %i'%epoch)
        print('Train')
        for _ in tqdm.tqdm(range(args.batches_per_epoch)):
            train_batch()
        
        checkpoint_path = './viewpoint_checkpoint_%04i.pt'%epoch
        print('Saving Checkpoint to: %s'%checkpoint_path)
        torch.save(viewpoint_model.state_dict(), checkpoint_path)
        
        #print('Test')
        #test_epoch(epoch)

else:
    test_epoch(1)
