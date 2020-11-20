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

import matplotlib.pyplot as pyplot

import segmentation_models_pytorch

import renderpy.masks as masks

import brick_gym.config as config
import brick_gym.evaluation as evaluation
from brick_gym.dataset.data_paths import data_paths
import brick_gym.dataset.ldraw_environment as ldraw_environment
import brick_gym.viewpoint.azimuth_elevation as azimuth_elevation
import brick_gym.random_stack.dataset as random_stack_dataset

# Read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
        '--encoder', type=str, default='se_resnext50_32x4d')#default='resnet34')
parser.add_argument(
        '--batch-size', type=int, default=4)
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
        '--edge-weight', type=float, default=1.0)
parser.add_argument(
        '--image-size', type=str, default='256x256')
parser.add_argument(
        '--checkpoint-frequency', type=int, default=10)
parser.add_argument(
        '--test-frequency', type=int, default=1)
parser.add_argument(
        '--test-split', type=str, default='test')
parser.add_argument(
        '--brick-vector-dimension', type=int, default=64)
parser.add_argument(
        '--dump-images', action='store_true')
args = parser.parse_args()

# Build the data generators
if not args.test:
    train_paths = data_paths(
            config.paths['random_stack'], 'train_mpd', subset=args.train_subset)

test_paths = data_paths(
        config.paths['random_stack'],
        '%s_mpd'%args.test_split,
        subset=args.test_subset)

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
        classes = args.brick_vector_dimension,
        activation = None).cuda()

brick_classifier = torch.nn.Conv2d(
        args.brick_vector_dimension, 7, 1).cuda()
confidence_classifier = torch.nn.Conv2d(
        args.brick_vector_dimension, 2, 1).cuda()

class BrickVectorEdgeModel(torch.nn.Module):
    def __init__(self):
        super(BrickVectorEdgeModel, self).__init__()
        
        self.linear_xy = torch.nn.Linear(2, 512)
        
        self.linear_a = torch.nn.Linear(args.brick_vector_dimension, 512)
        self.linear_b = torch.nn.Linear(512, 512)
        
        self.combination_a = torch.nn.Linear(1024, 512)
        self.combination_b = torch.nn.Linear(512, 512)
        self.combination_c = torch.nn.Linear(512, 512)
        self.edge_out = torch.nn.Linear(512,2)
        
    def forward(self, brick_vectors, xy):
        batch_size, bricks_per_model, _ = brick_vectors.shape
        brick_vectors = brick_vectors.view(-1, args.brick_vector_dimension)
        xy = xy.view(-1, 2)
        brick_features = self.linear_a(brick_vectors) + self.linear_xy(xy)
        brick_features = torch.nn.functional.relu(brick_features)
        brick_features = self.linear_b(brick_features)
        brick_features = torch.nn.functional.relu(brick_features)
        
        brick_features_a = brick_features.reshape(
                batch_size, 1, bricks_per_model, 512).expand(
                batch_size, bricks_per_model, bricks_per_model, 512)
        brick_features_b = brick_features.reshape(
                batch_size, bricks_per_model, 1, 512).expand(
                batch_size, bricks_per_model, bricks_per_model, 512)
        
        edge_features = torch.cat(
                (brick_features_a, brick_features_b), dim=-1).view(-1, 1024)
        edge_features = self.combination_a(edge_features)
        edge_features = torch.nn.functional.relu(edge_features)
        edge_features = self.combination_b(edge_features)
        edge_features = torch.nn.functional.relu(edge_features)
        edge_features = self.combination_c(edge_features)
        edge_features = torch.nn.functional.relu(edge_features)
        return self.edge_out(edge_features).view(
                batch_size, bricks_per_model, bricks_per_model, 2)

edge_classifier = BrickVectorEdgeModel().cuda()

# Build the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

#brick_weights = torch.FloatTensor([0.02, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4]).cuda()
brick_weights = torch.FloatTensor([0.02, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4]).cuda()
confidence_weights = torch.FloatTensor([1, 0.01]).cuda()
edge_weights = torch.FloatTensor([1.0, 1.0]).cuda()

def rollout(
        epoch,
        model_paths,
        num_episodes,
        steps_per_episode=8,
        ground_truth_foreground=True,
        mark_selection=False):
    model.eval()
    brick_classifier.eval()
    confidence_classifier.eval()
    edge_classifier.eval()
    with torch.no_grad():
        images = torch.zeros(
                num_episodes, steps_per_episode+1, 3, height, width)
        segmentation_targets = torch.zeros(
                num_episodes, steps_per_episode+1, height, width,
                dtype=torch.long)
        instance_ids = torch.zeros(
                num_episodes, steps_per_episode, dtype=torch.long)
        edge_targets = torch.zeros(
                num_episodes, steps_per_episode, steps_per_episode,
                dtype=torch.long)
        segmentation_logits = torch.zeros(
                num_episodes, steps_per_episode, 7, height, width)
        '''
        brick_vectors = torch.zeros(
                num_episodes,
                steps_per_episode,
                args.brick_vector_dimension)
        '''
        actions = torch.zeros(
                num_episodes, steps_per_episode, 2, dtype=torch.long)
        valid_entries = []
        for episode in tqdm.tqdm(range(num_episodes)):
            model_path = random.choice(model_paths)
            environment.load_path(model_path)
            image_numpy = environment.reset()
            if args.dump_images:
                Image.fromarray(image_numpy).save(
                        './image_%i_%i_0.png'%(epoch, episode))
            image = to_tensor(image_numpy)
            images[episode, 0] = image
            
            # super stupid to render twice for this...
            mask = environment.observe('mask')
            #instance_mask = environment.observe('instances')
            
            if args.dump_images:
                Image.fromarray(mask).save(
                        './mask_%i_%i_0.png'%(epoch, episode))
                #Image.fromarray(instance_mask).save(
                #        './instances_%i_%i_0.png'%(epoch, episode))
            segmentation_target = masks.color_byte_to_index(mask)
            segmentation_targets[episode, 0] = torch.LongTensor(
                    segmentation_target)
            #instance_id = masks.color_byte_to_index(instance_mask)
            #instance_ids[episode, 0] = torch.LongTensor(
            #        instance_id)
            #edge_target = 
            brick_order = []
            for step in range(steps_per_episode):
                # cudafy
                image = image.cuda().unsqueeze(0)
                
                # forward pass
                step_brick_vectors = model(image)
                #brick_vectors[episode, step] = step_brick_vectors
                
                # estimated segmentation
                batch_segmentation_logits = brick_classifier(step_brick_vectors)
                segmentation_logits[episode, step] = (
                        batch_segmentation_logits[0].cpu())
                segmentation_indices = torch.argmax(
                        batch_segmentation_logits, dim=1)
                if args.dump_images:
                    prediction_mask = masks.color_index_to_byte(
                            segmentation_indices[0].cpu().numpy())
                    Image.fromarray(prediction_mask).save(
                            './pred_%i_%i_%i.png'%(epoch, episode, step))
                
                # estimated confidence
                confidence_logits = confidence_classifier(step_brick_vectors)
                confidence = torch.softmax(confidence_logits, dim=1)[:,1]
                if args.dump_images:
                    conf = (confidence[0] * 255).type(torch.uint8).cpu().numpy()
                    Image.fromarray(conf).save(
                            './conf_%i_%i_%i.png'%(epoch, episode, step))
                    
                    conf_target = (
                            (segmentation_indices[0].cpu().numpy() ==
                             segmentation_target))
                    Image.fromarray((conf_target*255).astype(numpy.uint8)).save(
                            './conf_target_%i_%i_%i.png'%(epoch, episode, step))
                
                # select pixel-level action
                h, w = confidence.shape[-2:]
                if ground_truth_foreground:
                    foreground = torch.LongTensor(
                            segmentation_target != 0).cuda()
                else:
                    foreground = segmentation_indices != 0
                stop = int(torch.sum(foreground).cpu()) == 0
                if stop:
                    break
                foreground_confidence = confidence * foreground
                location = int(torch.argmax(foreground_confidence).cpu())
                y, x = numpy.unravel_index(location, (height, width))
                brick_name = environment.hide_brick_at_pixel(x, y)
                if brick_name is None:
                    brick_id = -1
                else:
                    brick_id = int(brick_name.split('_')[-1])-1
                brick_order.append(brick_id)
                
                if args.dump_images:
                    marked_numpy = image_numpy.copy()
                    marked_numpy[y-5:y+5,x] = (255,0,0)
                    marked_numpy[y,x-5:x+5] = (255,0,0)
                    Image.fromarray(marked_numpy).save(
                            './marked_%i_%i_%i.png'%(epoch, episode, step))
                
                # update data
                image_numpy = environment.observe('color')
                if args.dump_images:
                    Image.fromarray(image_numpy).save(
                            './image_%i_%i_%i.png'%(epoch, episode, step+1))
                image = to_tensor(image_numpy)
                images[episode, step+1] = image
                mask = environment.observe('mask')
                #instance_id = environment.observe('instances')
                if args.dump_images:
                    Image.fromarray(mask).save(
                            './mask_%i_%i_%i.png'%(epoch, episode, step+1))
                    #Image.fromarray(instance_ids).save(
                    #        './instances_%i_%i_%i.png'%(epoch, episode,step+1))
                segmentation_target = masks.color_byte_to_index(mask)
                segmentation_targets[episode, step+1] = torch.LongTensor(
                        segmentation_target)
                actions[episode, step, 0] = x
                actions[episode, step, 1] = y
                valid_entries.append((episode, step))
            brick_remap = {
                    brick_id : k for k, brick_id in enumerate(brick_order)
                    if brick_id != -1}
            for a,b in environment.edges:
                if a not in brick_remap or b not in brick_remap:
                    continue
                aa = brick_remap[a]
                bb = brick_remap[b]
                edge_targets[episode,aa,bb] = 1
                edge_targets[episode,bb,aa] = 1
        
        return (images,
                segmentation_targets,
                edge_targets,
                segmentation_logits,
                actions,
                valid_entries)

# Train an epoch
def train_epoch(epoch):
    print('Train Epoch: %i'%epoch)
    
    # generate some data with the latest model
    print('Generating Data')
    images, segmentation_targets, edge_targets, _, actions, valid_entries = (
            rollout(epoch, train_paths, args.episodes_per_train_epoch))
    valid_entries = set(valid_entries)
    
    
    model.train()
    brick_classifier.train()
    confidence_classifier.train()
    edge_classifier.train()
    
    
    # train on the newly generated data
    for mini_epoch in range(1, args.num_mini_epochs+1):
        print('Train Mini Epoch: %i'%mini_epoch)
        num_episodes, num_steps = actions.shape[:2]
        episode_order = list(range(num_episodes))
        random.shuffle(episode_order)
        train_iterate = tqdm.tqdm(range(0, num_episodes, args.batch_size))
        for i in train_iterate:
            episodes = episode_order[i:i+args.batch_size]
            batch_size = len(episodes)
            selected_brick_vectors = []
            valid_edge_entries = torch.ones(
                    batch_size, num_steps, num_steps).cuda()
            
            episode_class_loss = 0
            episode_confidence_loss = 0
            for j in range(num_steps):
                batch_valid_entries = torch.FloatTensor(
                        [(episode, j) in valid_entries for episode in episodes])
                batch_valid_entries = batch_valid_entries.cuda()
                valid_edge_entries[:,j,:] *= batch_valid_entries.unsqueeze(1)
                valid_edge_entries[:,:,j] *= batch_valid_entries.unsqueeze(1)
                
                batch_valid_entries = batch_valid_entries.view(
                        len(episodes), 1, 1)
                
                batch_images = images[episodes, j].cuda()
                batch_segmentation_targets = (
                        segmentation_targets[episodes, j].cuda())
                
                brick_vectors = model(batch_images)
                step_actions = actions[episodes, j]
                x = step_actions[:,0]
                y = step_actions[:,1]
                selected_brick_vectors.append(
                        brick_vectors[range(batch_size),:,y,x])
                class_logits = brick_classifier(brick_vectors)
                confidence_logits = confidence_classifier(brick_vectors)
                
                # brick class loss
                class_loss = torch.nn.functional.cross_entropy(
                        class_logits, batch_segmentation_targets,
                        weight=brick_weights, reduction='none')
                class_loss = class_loss * batch_valid_entries
                #class_loss = torch.mean(class_loss)
                divisor = torch.sum(batch_valid_entries)
                if divisor:
                    class_loss = torch.sum(class_loss) / divisor.float()
                    class_loss = class_loss / (height * width)
                else:
                    class_loss = 0.
                episode_class_loss = episode_class_loss + class_loss
                
                # confidence loss
                predictions = torch.argmax(class_logits, dim=1).detach()
                confidence_target = (predictions == batch_segmentation_targets)
                
                confidence_loss = torch.nn.functional.cross_entropy(
                        confidence_logits,
                        confidence_target.long(),
                        weight = confidence_weights,
                        reduction = 'none')
                confidence_loss = confidence_loss * batch_valid_entries
                #confidence_loss = torch.mean(confidence_loss)
                if divisor:
                    confidence_loss = (
                            torch.sum(confidence_loss) / divisor.float())
                    confidence_loss = confidence_loss / (height * width)
                else:
                    confidence_loss = 0.
                episode_confidence_loss = (
                        episode_confidence_loss + confidence_loss)
                
                # combine loss
                #loss = (class_loss * args.class_weight +
                #        confidence_loss * args.confidence_weight)
                
                # backprop
                #loss.backward()
                #optimizer.step()
                #optimizer.zero_grad()
            
            # edge-prediction forward pass
            selected_brick_vectors = torch.stack(selected_brick_vectors, 1)
            #selected_locations = torch.stack(selected_locations, 1)
            batch_actions = actions[episodes].detach().float().cuda()
            batch_actions[:,:,0] /= float(width)
            batch_actions[:,:,1] /= float(height)
            edge_logits = edge_classifier(selected_brick_vectors, batch_actions)
            batch_size, steps, _, _ = edge_logits.shape
            
            # edge-prediction target (bs, steps, steps)
            batch_edge_targets = edge_targets[episodes].cuda()
            edge_loss = torch.nn.functional.cross_entropy(
                    edge_logits.view(-1,2),
                    batch_edge_targets.view(-1),
                    weight = edge_weights,
                    reduction = 'none').view(batch_size, steps, steps)
            edge_loss = edge_loss * valid_edge_entries
            divisor = torch.sum(valid_edge_entries)
            if divisor:
                edge_loss = torch.sum(edge_loss) / divisor
            else:
                edge_loss = 0.
            
            # sum loss
            loss = (episode_class_loss/num_steps * args.class_weight +
                    episode_confidence_loss/num_steps * args.confidence_weight +
                    edge_loss * args.edge_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # do that forward pass
            # and backward pass
            train_iterate.set_description(
                    'S: %.04f C: %.04f E: %.04f'%(
                        float(episode_class_loss)/num_steps * args.class_weight,
                        float(episode_confidence_loss)/num_steps *
                            args.confidence_weight,
                        float(edge_loss)/num_steps * args.edge_weight))

# Test an epoch
def test_epoch(epoch, mark_selection=False, graph=False):
    print('Test Epoch: %i'%epoch)
    print('Generating Data')
    (images,
     segmentation_targets,
     edge_targets,
     segmentation_logits,
     actions,
     valid_entries) = rollout(
            epoch,
            test_paths,
            args.episodes_per_test_epoch,
            ground_truth_foreground = False,
            mark_selection = mark_selection)
    episodes, steps = zip(*valid_entries)
    valid_images = images[episodes, steps]
    valid_targets = segmentation_targets[episodes, steps]
    valid_logits = segmentation_logits[episodes, steps]
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
        pixel_logits = segmentation_logits[episode, step, :, y, x]
        prediction = torch.argmax(pixel_logits)
        target = segmentation_targets[episode, step, y, x]
        step_correct += int(prediction == target)
        iterate.set_description('Acc: %f'%(step_correct/(i+1)))
    
    print('Step Accuracy: %f'%(step_correct/len(valid_entries)))
    
    with torch.no_grad():
        num_episodes, num_steps = actions.shape[:2]
        episode_order = list(range(num_episodes))
        edge_true_positive = 0
        edge_false_negative = 0
        edge_false_positive = 0
        edge_scores = []
        edge_ground_truth = []
        #edge_total = 0
        test_iterate = tqdm.tqdm(range(0, num_episodes, args.batch_size))
        for i in test_iterate:
            episodes = episode_order[i:i+args.batch_size]
            batch_size = len(episodes)
            selected_brick_vectors = []
            valid_edge_entries = torch.ones(
                    batch_size, num_steps, num_steps).cuda()
            
            episode_class_loss = 0
            episode_confidence_loss = 0
            for j in range(num_steps):
                batch_valid_entries = torch.FloatTensor(
                        [(episode, j) in valid_entries for episode in episodes])
                batch_valid_entries = batch_valid_entries.cuda()
                valid_edge_entries[:,j,:] *= batch_valid_entries.unsqueeze(1)
                valid_edge_entries[:,:,j] *= batch_valid_entries.unsqueeze(1)
                
                batch_valid_entries = batch_valid_entries.view(
                        len(episodes), 1, 1, 1)
                
                batch_images = images[episodes, j].cuda()
                batch_segmentation_targets = (
                        segmentation_targets[episodes, j].cuda())
                
                brick_vectors = model(batch_images)
                step_actions = actions[episodes, j]
                x = step_actions[:,0]
                y = step_actions[:,1]
                selected_brick_vectors.append(
                        brick_vectors[range(batch_size),:,y,x])
            
            # edge-prediction forward pass
            selected_brick_vectors = torch.stack(selected_brick_vectors, 1)
            batch_actions = actions[episodes].detach().float().cuda()
            batch_actions[:,:,0] /= float(width)
            batch_actions[:,:,1] /= float(height)
            edge_logits = edge_classifier(selected_brick_vectors, batch_actions)
            batch_size, steps, _, _ = edge_logits.shape
            
            # edge-prediction target (bs, steps, steps)
            batch_edge_targets = edge_targets[episodes].cuda()
            #batch_edge_prediction = torch.argmax(edge_logits, dim=-1)
            for j in range(num_steps-1):
                for k in range(j+1, num_steps):
                    logits = edge_logits[:,j,k]
                    edge_probability = torch.softmax(logits, dim=-1)[:,-1]
                    edge_scores.extend(edge_probability.cpu().tolist())
                    edge_target = batch_edge_targets[:,j,k]
                    edge_ground_truth.extend(edge_target.cpu().tolist())
                    #logits_b = edge_logits[:,k,j]
                    #logits = logits_a + logits_b
                    '''
                    edge_prediction = torch.argmax(logits, dim=-1)
                    edge_target = batch_edge_targets[:,j,k]
                    tp, fp, fn = evaluation.tp_fp_fn(
                            edge_prediction.cpu().numpy(),
                            edge_target.cpu().numpy())
                    edge_true_positive += int(numpy.sum(tp))
                    edge_false_positive += int(numpy.sum(fp))
                    edge_false_negative += int(numpy.sum(fn))
                    '''
            
            '''
            if i == 0:
                for j in range(4):
                    print('Edge Targets:')
                    print(batch_edge_targets[j].cpu())
                    print('Edge Prediction:')
                    print(batch_edge_prediction[j].cpu())
                    print('Valid Edge Entries:')
                    print(valid_edge_entries[j])
            '''
            #batch_edge_correct = batch_edge_prediction == batch_edge_targets
            #batch_edge_correct = batch_edge_correct * valid_edge_entries
            #edge_correct += float(torch.sum(batch_edge_correct))
            #edge_total += float(torch.sum(valid_edge_entries))
            #test_iterate.set_description(
            #        'Edge Acc: %.04f'%(edge_correct/edge_total))
        #print('Final Edge Accuracy: %.04f'%(edge_correct/edge_total))
        '''
        p, r = evaluation.precision_recall(
                edge_true_positive, edge_false_positive, edge_false_negative)
        f1 = evaluation.f1(p, r)
        print('Edge Precision: %f'%p)
        print('Edge Recall: %f'%r)
        print('Edge F1: %f'%f1)
        '''
        pr, concave_pr, ap = evaluation.ap(
                edge_scores, edge_ground_truth, 0)
        concave_pr.append([0,1])
        
        '''
        random_pr, random_concave_pr, random_ap = evaluation.ap(
                [random.random() for _ in edge_scores], edge_ground_truth, 0)
        random_concave_pr.append([0,1])
        '''
        
        print('AP: %f'%ap)
        edge_density = sum(edge_ground_truth) / len(edge_ground_truth)
        print('Edge Density: %f'%edge_density)
        if graph:
            x, y = zip(*concave_pr)
            pyplot.plot(x, y, label='ours')
            
            #x2, y2 = zip(*random_concave_pr)
            #pyplot.plot(x2, y2)
            
            random_concave_pr = [
                    [0,1], [edge_density, 1], [edge_density, 0], [1,0]]
            x2, y2 = zip(*random_concave_pr)
            pyplot.plot(x2, y2, label='random')
            
            pyplot.legend()
            pyplot.savefig('./ap_%i.png'%epoch)

if args.test:
    model_checkpoint = './model_checkpoint_%04i.pt'%args.num_epochs
    model.load_state_dict(torch.load(model_checkpoint))
    
    segmentation_checkpoint = (
            './segmentation_checkpoint_%04i.pt'%args.num_epochs)
    brick_classifier.load_state_dict(torch.load(segmentation_checkpoint))
    
    confidence_checkpoint = (
            './confidence_checkpoint_%04i.pt'%args.num_epochs)
    confidence_classifier.load_state_dict(torch.load(confidence_checkpoint))
    
    edge_checkpoint = (
            './edge_checkpoint_%04i.pt'%args.num_epochs)
    edge_classifier.load_state_dict(torch.load(edge_checkpoint))
    
    test_epoch(args.num_epochs, mark_selection=True, graph=True)

else:
    for epoch in range(1, args.num_epochs+1):
        print('='*80)
        t0 = time.time()
        # train
        train_epoch(epoch)
        
        # save
        if epoch % args.checkpoint_frequency == 0:
            print('-'*80)
            model_checkpoint_path = './model_checkpoint_%04i.pt'%epoch
            print('Saving main model checkpoint to: %s'%model_checkpoint_path)
            torch.save(model.state_dict(), model_checkpoint_path)
            
            segmentation_checkpoint_path = (
                    './segmentation_checkpoint_%04i.pt'%epoch)
            print('Saving segmentation checkpoint to: %s'%
                    segmentation_checkpoint_path)
            torch.save(brick_classifier.state_dict(),
                    segmentation_checkpoint_path)
            
            confidence_checkpoint_path = (
                    './confidence_checkpoint_%04i.pt'%epoch)
            print('Saving segmentation checkpoint to: %s'%
                    segmentation_checkpoint_path)
            torch.save(confidence_classifier.state_dict(),
                    confidence_checkpoint_path)
            
            edge_checkpoint_path = (
                    './edge_checkpoint_%04i.pt'%epoch)
            print('Saving edge checkpoint to: %s'%edge_checkpoint_path)
            torch.save(edge_classifier.state_dict(), edge_checkpoint_path)
        
        # test
        if epoch % args.test_frequency == 0:
            print('-'*80)
            test_epoch(epoch)
        print('Elapsed: %.04f'%(time.time() - t0))
