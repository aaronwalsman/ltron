#!/usr/bin/env python
import random
import argparse

import numpy

import tqdm

import torch
from torchvision.transforms.functional import to_tensor

import PIL.Image as Image

import moviepy.editor

import segmentation_models_pytorch

import renderpy.masks as masks

import brick_gym.config as config
import brick_gym.mpd_sequence as mpd_sequence
import brick_gym.viewpoint_control as viewpoint_control

def seed_everything():
    random.seed(1234)
    numpy.random.seed(1234)
    torch.manual_seed(1234)
seed_everything()

parser = argparse.ArgumentParser()
parser.add_argument(
        '--segmentation-model', type=str, default='resnet18')
parser.add_argument(
        '--segmentation-checkpoint', type=str)
parser.add_argument(
        '--confidence-model', type=str, default='resnet18')
parser.add_argument(
        '--confidence-checkpoint', type=str)
parser.add_argument(
        '--num-examples', type=int, default=1000)
parser.add_argument(
        '--render-examples', type=int, default=16)
args = parser.parse_args()

segmentation_model = segmentation_models_pytorch.FPN(
        encoder_name = args.segmentation_model,
        encoder_weights = None,
        classes = 7,
        activation = None).cuda()
state_dict = torch.load(args.segmentation_checkpoint)
segmentation_model.load_state_dict(state_dict)
segmentation_model.eval()

confidence_model = segmentation_models_pytorch.FPN(
        encoder_name = args.confidence_model,
        encoder_weights = None,
        classes = 2,
        activation = None).cuda()
state_dict = torch.load(args.confidence_checkpoint)
confidence_model.load_state_dict(state_dict)
confidence_model.eval()

test_view_control = viewpoint_control.AzimuthElevationViewpointControl(
        reset_mode = 'random',
        elevation_range = [-1.0, 1.0])
test_mpd_sequence = mpd_sequence.MPDSequence(
        test_view_control,
        directory = config.paths['random_stack'],
        split = 'test')

with torch.no_grad():
    sequence_ps = []
    sequence_rs = []
    iterate = tqdm.tqdm(range(args.num_examples))
    for i in iterate:
        test_mpd_sequence.reset()
        sequence_tp = 0
        sequence_fp = 0
        sequence_fn = 0
        sequence_images = []
        for j in range(8):
            image, mask = test_mpd_sequence.observe()
            x = to_tensor(image).unsqueeze(0).cuda()
            segmentation_logits = segmentation_model(x)
            segmentation_indices = torch.argmax(segmentation_logits, dim=1)[0]
            confidence_logits = confidence_model(x)
            confidence = torch.softmax(confidence_logits, dim=1)[0,1]
            height, width = confidence.shape[-2:]
            predicted_foreground = segmentation_indices != 0
            stop = int(torch.sum(predicted_foreground).cpu()) == 0
            if stop:
                break
            
            foreground_confidence = confidence * predicted_foreground
            location = int(torch.argmax(foreground_confidence).cpu())
            y, x = numpy.unravel_index(location, (height, width))
            test_mpd_sequence.hide_brick_at_pixel(x, y)
            
            mask_indices = masks.color_byte_to_index(mask)
            #print(mask_indices.shape)
            #print(segmentation_indices.shape)
            match = (mask_indices[y,x] == segmentation_indices[y,x]).cpu()
            sequence_tp += int(match)
            sequence_fp += 1 - int(match)
            
            predicted_mask = masks.color_index_to_byte(
                    segmentation_indices.cpu())
            
            image = image.copy()
            image[y,x-5:x+5] = (255,0,0)
            image[y-5:y+5,x] = (255,0,0)
            if i < args.render_examples:
                sequence_images.append(image)
                Image.fromarray(image).save('./image_%i_%i.png'%(i,j))
                Image.fromarray(predicted_mask).save(
                        './prediction_%i_%i.png'%(i,j))
                Image.fromarray(mask).save('./mask_%i_%i.png'%(i,j))
                confidence_image = (confidence * 255).cpu().numpy().astype(
                        numpy.uint8)
                Image.fromarray(confidence_image).save(
                        './confidence_%i_%i.png'%(i,j))
        
        image_clip = moviepy.editor.ImageSequenceClip(
                sequence_images, fps=2)
        image_clip.write_gif(
                './image_seq_%i.gif'%i, fps=2)
        sequence_fn = (
                len(test_mpd_sequence.renderer.list_instances()) - sequence_tp)
        
        if sequence_tp + sequence_fp == 0:
            sequence_p = 0
        else:
            sequence_p = sequence_tp / (sequence_tp + sequence_fp)
        sequence_r = sequence_tp / (sequence_tp + sequence_fn)
        sequence_ps.append(sequence_p)
        sequence_rs.append(sequence_r)
        iterate.set_description('P: %.04f, R: %.04f'%(sequence_p, sequence_r))
    
    print('Average sequence Precision: %.04f'%(
            sum(sequence_ps)/len(sequence_ps)))
    print('Average sequence Recall: %.04f'%(
            sum(sequence_rs)/len(sequence_rs)))
