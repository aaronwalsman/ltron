#!/usr/bin/env python
import time
import random
import math
import argparse

import torch
from torchvision.transforms.functional import to_tensor
from torchvision.models import resnet34, resnet18

import numpy

import PIL.Image as Image

import tqdm

import matplotlib.pyplot as pyplot

import segmentation_models_pytorch

import renderpy.masks as masks

import brick_gym.config as config
import brick_gym.utils as utils
import brick_gym.evaluation as evaluation
from brick_gym.dataset.paths import get_dataset_paths
import brick_gym.dataset.ldraw_environment as ldraw_environment
import brick_gym.viewpoint.azimuth_elevation as azimuth_elevation
import brick_gym.dataset.old_dataset as random_stack_dataset
import brick_gym.torch.models.resnet as bg_resnet
import brick_gym.torch.models.standard_models as standard_models
import brick_gym.torch.utils as bgt_utils
from brick_gym.envs.graph_env import GraphEnv
from brick_gym.multiclass import MultiClass

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
        '--class-weight', type=float, default=0.8)
parser.add_argument(
        '--confidence-weight', type=float, default=0.2)
parser.add_argument(
        '--edge-weight', type=float, default=0.1)
parser.add_argument(
        '--image-size', type=str, default='256x256')
parser.add_argument(
        '--checkpoint-frequency', type=int, default=10)
parser.add_argument(
        '--test-frequency', type=int, default=1)
parser.add_argument(
        '--test-split', type=str, default='test')
parser.add_argument(
        '--brick-vector-dimension', type=int, default=512)
parser.add_argument(
        '--num-processes', type=int, default=16)
parser.add_argument(
        '--dump-images', action='store_true')
parser.add_argument(
        '--accumulator-mode', type=str, default='argmax')
args = parser.parse_args()

# Build the data generators
if not args.test:
    train_paths = get_dataset_paths(
            #config.datasets['random_stack'],
            'random_stack',
            'train_mpd',
            subset=args.train_subset)

test_paths = get_dataset_paths(
        #config.paths['random_stack'],
        'random_stack',
        '%s_mpd'%args.test_split,
        subset=args.test_subset)

width, height = args.image_size.split('x')
width = int(width)
height = int(height)
max_bricks_per_scene = 8
viewpoint_control = azimuth_elevation.FixedAzimuthalViewpoint(
        azimuth = math.radians(30), elevation = -math.radians(45))
multi_environment = ldraw_environment.MultiLDrawEnvironment(
        num_processes = args.num_processes,
        width = width,
        height = height,
        viewpoint_control = viewpoint_control)

# Build the model
model = resnet18(pretrained=True).cuda()
bg_resnet.make_spatial_attention_resnet(
        model, shape=(height, width), do_spatial_embedding=True)
bg_resnet.replace_fc(model, args.brick_vector_dimension)
brick_classifier = torch.nn.Linear(
        args.brick_vector_dimension, 7).cuda()
confidence_classifier = torch.nn.Linear(
        args.brick_vector_dimension, 2).cuda()

'''
class BrickVectorEdgeModel(torch.nn.Module):
    def __init__(self):
        super(BrickVectorEdgeModel, self).__init__()
        
        self.linear_a = torch.nn.Linear(args.brick_vector_dimension, 512)
        self.linear_b = torch.nn.Linear(512, 512)
        
        self.combination_a = torch.nn.Linear(1024, 512)
        self.combination_b = torch.nn.Linear(512, 512)
        self.combination_c = torch.nn.Linear(512, 512)
        self.edge_out = torch.nn.Linear(512,2)
        
    def forward(self, brick_vectors):
        batch_size, bricks_per_model, _ = brick_vectors.shape
        brick_vectors = brick_vectors.view(-1, args.brick_vector_dimension)
        brick_features = self.linear_a(brick_vectors)
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
'''

edge_classifier = standard_models.get_edge_model('simple_edge_512').cuda()

# Build the optimizer
optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(brick_classifier.parameters()) +
        list(confidence_classifier.parameters()) +
        list(edge_classifier.parameters()), lr=3e-4)

confidence_weights = torch.FloatTensor([1, 0.01]).cuda()
edge_weights = torch.FloatTensor([1.0, 1.0]).cuda()

def get_instance_class_lookup(instance_brick_types):
    lookup = numpy.zeros(1 + max_bricks_per_scene, dtype=numpy.long)
    for instance_id, mesh_name in instance_brick_types.items():
        lookup[instance_id] = mesh_indices[mesh_name]
    return lookup

def rollout(
        epoch,
        model_paths,
        num_episodes,
        ground_truth_foreground,
        steps_per_episode=8,
        mark_selection=False):
    model.eval()
    brick_classifier.eval()
    confidence_classifier.eval()
    edge_classifier.eval()
    with torch.no_grad():
        masked_images = torch.zeros(
                num_episodes, steps_per_episode+1, max_bricks_per_scene+1,
                3, height, width)
        class_targets = torch.zeros(
                num_episodes, steps_per_episode+1, max_bricks_per_scene+1,
                dtype = torch.long)
        edge_targets = torch.zeros(
                num_episodes, max_bricks_per_scene, max_bricks_per_scene,
                dtype=torch.long)
        logits = torch.zeros(
                num_episodes, steps_per_episode, max_bricks_per_scene+1, 7+2)
        actions = torch.zeros(
                num_episodes,
                steps_per_episode,
                dtype=torch.long)
        episode_class_targets = torch.zeros(
                num_episodes, max_bricks_per_scene, dtype=torch.long)
        
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
            for i, lookup in enumerate(instance_class_lookup):
                episode = ep_start + i
                episode_class_targets[episode] = torch.LongTensor(lookup[1:])
            observations = multi_environment.observe(
                    ('color', 'instance_labels'))
            for i, (image, mask_indices) in enumerate(observations):
                episode = ep_start + i
                if args.dump_images:
                    Image.fromarray(image).save(
                            './image_%i_%i_0.png'%(epoch, episode))
                image = to_tensor(image)
                #mask_indices = masks.color_byte_to_index(mask)
                for j in range(1, max_bricks_per_scene+1):
                    brick_mask = torch.FloatTensor(mask_indices == j)
                    if torch.sum(brick_mask):
                        masked_image = image * brick_mask
                        masked_images[episode, 0, j] = masked_image
                        class_targets[episode, 0, j] = (
                                instance_class_lookup[i][j])
                        
                        if args.dump_images:
                            image_numpy = masked_image.permute(1,2,0).numpy()
                            image_numpy = (image_numpy * 255).astype(
                                    numpy.uint8)
                            Image.fromarray(image_numpy).save(
                                    './masked_image_%i_%i_0_%i.png'%(
                                        epoch, episode, j))
            valid_entries.extend(
                    [(ep_start+b,0) for b in range(len(observations))])
            
            brick_order = []
            for step in range(steps_per_episode):
                # cudafy
                images = masked_images[ep_start:ep_end, step].cuda()
                bs, num_bricks, _, h, w = images.shape
                images = images.view(-1, 3, height, width)
                batch_class_targets = (
                        class_targets[ep_start:ep_end, step].cuda())
                
                # forward pass
                step_brick_vectors = model(images)
                
                # estimated segmentation
                batch_class_logits = brick_classifier(
                        step_brick_vectors)
                batch_class_logits = batch_class_logits.view(bs, num_bricks, -1)
                logits[ep_start:ep_end, step, :, :7] = batch_class_logits.cpu()
                batch_class_prediction = torch.argmax(batch_class_logits, dim=2)
                
                # estimated confidence
                batch_confidence_logits = confidence_classifier(
                        step_brick_vectors)
                batch_confidence_logits = batch_confidence_logits.view(
                        bs, num_bricks, 2)
                logits[ep_start:ep_end, step, :, 7:] = (
                        batch_confidence_logits.cpu())
                confidence = torch.softmax(
                        batch_confidence_logits, dim=-1)[:,:,1]
                
                # select brick-level action
                if ground_truth_foreground:
                    foreground = (batch_class_targets != 0).long()
                else:
                    foreground = batch_class_prediction != 0
                stop = torch.sum(foreground, dim=1).cpu() == 0
                
                foreground_confidence = confidence * foreground
                location = torch.argmax(foreground_confidence, dim=1)
                valid_locations = batch_class_targets[
                        range(batch_class_targets.shape[0]),location] != 0
                valid_locations = valid_locations.long()
                location = location * valid_locations
                actions[ep_start:ep_end,step] = location
                hide_indices = location.tolist()
                multi_environment.hide_bricks(hide_indices)
                
                brick_order.append(hide_indices)
                
                previous_observations = observations
                observations = multi_environment.observe(
                        ('color', 'instance_labels'))
                for i, (image, mask_indices) in enumerate(observations):
                    episode = ep_start + i
                    if args.dump_images:
                        Image.fromarray(image).save(
                                './image_%i_%i_%i.png'%(epoch, episode, step+1))
                    image = to_tensor(image)
                    #mask_indices = masks.color_byte_to_index(mask)
                    for j in range(1, max_bricks_per_scene+1):
                        brick_mask = torch.FloatTensor(mask_indices == j)
                        if torch.sum(brick_mask):
                            masked_image = image * brick_mask
                            masked_images[episode, step+1, j] = masked_image
                            class_targets[episode, step+1, j] = (
                                    instance_class_lookup[i][j])
                            
                            if args.dump_images:
                                image_numpy = masked_image.permute(
                                        1,2,0).numpy()
                                image_numpy = (image_numpy * 255).astype(
                                        numpy.uint8)
                                Image.fromarray(image_numpy).save(
                                        './masked_image_%i_%i_%i_%i.png'%(
                                            epoch, episode, step+1, j))
                                if j == location[i]:
                                    im = Image.open(
                                            './masked_image_%i_%i_%i_%i.png'%(
                                            epoch, episode, step, j))
                                    im.save(
                                            './selected_image_%i_%i_%i.png'%(
                                            epoch, episode, step))
                
                valid_entries.extend([
                        (ep_start + i, step)
                        for i in range(len(observations))
                        if not stop[i]])
                
            batch_edges = multi_environment.get_edges()
            for i, edges in enumerate(batch_edges):
                episode = ep_start + i
                bo = [order[i] for order in brick_order]
                brick_remap = {
                        brick_id : k
                        for k, brick_id in enumerate(bo)
                        if brick_id != 0}
                for a, b in edges:
                    try:
                        aa = brick_remap[a+1]
                        bb = brick_remap[b+1]
                    except KeyError:
                        print('unable to get aa, bb')
                        print(bo)
                        print(brick_remap)
                    edge_targets[episode,aa,bb] = 1
                    edge_targets[episode,bb,aa] = 1
        
        return (masked_images,
                class_targets,
                logits,
                edge_targets,
                actions,
                valid_entries,
                episode_class_targets)

# Train an epoch
def train_epoch(epoch):
    print('Train Epoch: %i'%epoch)
    
    # generate some data with the latest model
    print('Generating Data')
    (images,
     class_targets,
     logits,
     edge_targets,
     actions,
     valid_entries,
     episode_class_targets) = rollout(
                epoch, train_paths, args.episodes_per_train_epoch,
                ground_truth_foreground=True)
    num_episodes, num_steps = actions.shape[:2]
    valid_entries = set(valid_entries)
    
    model.train()
    brick_classifier.train()
    confidence_classifier.train()
    edge_classifier.train()
    
    # train on the newly generated data
    for mini_epoch in range(1, args.num_mini_epochs+1):
        print('Train Mini Epoch: %i'%mini_epoch)
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
                        len(episodes), 1)
                
                batch_images = images[episodes, j].cuda()
                bs, bt = batch_images.shape[:2]
                batch_class_targets = class_targets[episodes, j].cuda()
                
                brick_vectors = model(batch_images.view(-1, 3, height, width))
                brick_vectors_reshape = brick_vectors.view(bs, bt, -1)
                step_actions = actions[episodes, j]
                selected_brick_vectors.append(
                        brick_vectors_reshape[range(batch_size),step_actions])
                class_logits = brick_classifier(brick_vectors)
                confidence_logits = confidence_classifier(brick_vectors)
                
                # brick class loss
                class_loss = torch.nn.functional.cross_entropy(
                        class_logits, batch_class_targets.view(-1),
                        reduction='none')
                class_loss = class_loss.view(bs, bt)
                class_loss = class_loss * batch_valid_entries
                #class_loss = torch.mean(class_loss)
                divisor = torch.sum(batch_valid_entries) * bt
                if divisor:
                    class_loss = torch.sum(class_loss) / divisor.float()
                else:
                    class_loss = 0.
                episode_class_loss = episode_class_loss + class_loss
                
                # confidence loss
                predictions = torch.argmax(class_logits, dim=1).detach()
                confidence_target = predictions == batch_class_targets.view(-1)
                
                confidence_loss = torch.nn.functional.cross_entropy(
                        confidence_logits,
                        confidence_target.long(),
                        weight = confidence_weights,
                        reduction = 'none')
                confidence_loss = confidence_loss.view(bs, bt)
                confidence_loss = confidence_loss * batch_valid_entries
                if divisor:
                    confidence_loss = (
                            torch.sum(confidence_loss) / divisor.float())
                else:
                    confidence_loss = 0.
                episode_confidence_loss = (
                        episode_confidence_loss + confidence_loss)
            
            # edge-prediction forward pass
            selected_brick_vectors = torch.stack(selected_brick_vectors, 1)
            edge_logits = edge_classifier(selected_brick_vectors)
            #batch_size, steps, _, _ = edge_logits.shape
            batch_size, steps, _ = edge_logits.shape
            
            # edge-prediction target (bs, steps, steps)
            batch_edge_targets = edge_targets[episodes].cuda()
            edge_loss = torch.nn.functional.binary_cross_entropy(
                    torch.sigmoid(edge_logits.view(-1)),
                    batch_edge_targets.view(-1).float(),
                    reduction = 'none').view(batch_size, steps, steps)
            '''
            edge_loss = torch.nn.functional.cross_entropy(
                    edge_logits.view(-1,2),
                    batch_edge_targets.view(-1),
                    weight = edge_weights,
                    reduction = 'none').view(batch_size, steps, steps)
            '''
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
     class_targets,
     logits,
     edge_targets,
     actions,
     valid_entries,
     episode_class_targets) = rollout(
            epoch,
            test_paths,
            args.episodes_per_test_epoch,
            ground_truth_foreground = False,
            mark_selection = mark_selection)
    if not len(valid_entries):
        print('No good data yet')
        return
    
    episodes, steps = zip(*valid_entries)
    valid_images = images[episodes, steps]
    valid_class_targets = class_targets[episodes, steps]
    valid_logits = logits[episodes, steps]
    valid_actions = actions[episodes, steps]
    #valid_logits = class_logits[episodes, steps]
    #batch_size, steps, c, h, w = images.shape
    #images = images.view(batch_size * steps, c, h, w)
    #targets = targets.view(batch_size * steps, c, h, w)
    #logits = logits.view(batch_size * steps, c, h, w)
    
    #class_logits = valid_logits[:,:-2]
    #confidence_logits = valid_logits[:,-2:]
    
    #class_predictions = torch.argmax(class_logits, dim=1)
    #seg_correct = valid_actions == valid_class_targets
    #seg_total = valid_class_targets.numel()
    
    class_logits = valid_logits[:,:,:7]
    predictions = torch.argmax(class_logits, dim=-1)
    class_correct = valid_class_targets == predictions
    class_total = torch.numel(class_correct)
    
    print('Accuracy: %f'%(float(torch.sum(class_correct)) / class_total))
    
    foreground = (valid_class_targets != 0).long()
    class_correct_foreground = class_correct * foreground
    print('Foreground Accuracy: %f'%(
            float(torch.sum(class_correct_foreground)) /
            float(torch.sum(foreground))))
    print('Foreground Ratio: %f'%(
            float(torch.sum(foreground))/torch.numel(foreground)))
    
    step_correct = 0
    iterate = tqdm.tqdm(valid_entries)
    for i, (episode, step) in enumerate(iterate):
        action = actions[episode, step]
        #prediction = predictions[i][action]
        step_correct += int(class_correct[i][action])
        '''
        pixel_logits = segmentation_logits[episode, step, :, y, x]
        prediction = torch.argmax(pixel_logits)
        target = segmentation_targets[episode, step, y, x]
        step_correct += int(prediction == target)
        iterate.set_description('Acc: %f'%(step_correct/(i+1)))
        '''
    
    print('Step Accuracy: %f'%(step_correct/len(valid_entries)))
    
    with torch.no_grad():
        num_episodes, num_steps = actions.shape[:2]
        episode_order = list(range(num_episodes))
        #edge_scores = []
        #edge_ground_truth = []
        #edge_total = 0
        test_iterate = tqdm.tqdm(range(0, num_episodes, args.batch_size))
        all_predicted_edges = {}
        all_ground_truth_edges = {}
        for i in test_iterate:
            episodes = episode_order[i:i+args.batch_size]
            batch_size = len(episodes)
            selected_brick_vectors = []
            node_predictions = []
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
                        len(episodes), 1)
                
                batch_images = images[episodes, j].cuda()
                bs, bt = batch_images.shape[:2]
                batch_class_targets = (
                        class_targets[episodes, j].cuda())
                
                brick_vectors = model(batch_images.view(-1, 3, height, width))
                brick_vectors_reshape = brick_vectors.view(bs, bt, -1)
                step_actions = actions[episodes, j]
                selected_brick_vectors.append(
                        brick_vectors_reshape[range(batch_size),step_actions])
                
                if args.dump_images:
                    for k in range(batch_size):
                        selected_image = batch_images[k, step_actions[k]]
                        selected_image = (selected_image * 255).cpu().type(
                                torch.uint8).permute(1,2,0).numpy()
                        Image.fromarray(selected_image).save(
                                './verify_selected_image_%i_%i_%i.png'%(
                                epoch, i+k, j))
            
            # edge-prediction forward pass
            selected_brick_vectors = torch.stack(selected_brick_vectors, 1)
            edge_logits = edge_classifier(selected_brick_vectors)
            batch_size, steps, _ = edge_logits.shape
            
            # edge-prediction target (bs, steps, steps)
            batch_edge_targets = edge_targets[episodes].cuda()
            
            #batch_edge_prediction = torch.argmax(edge_logits, dim=-1)
            '''
            for j in range(num_steps-1):
                for k in range(j+1, num_steps):
                    logits = edge_logits[:,j,k]
                    #edge_probability = torch.softmax(logits, dim=-1)[:,-1]
                    edge_probability = torch.sigmoid(logits)
                    edge_scores.extend(edge_probability.cpu().tolist())
                    edge_target = batch_edge_targets[:,j,k]
                    edge_ground_truth.extend(edge_target.cpu().tolist())
            '''
            
            # bs, steps, bricks+1, class(7) + conf(2)
            # print(logits.shape)
            # bs, steps
            # print(actions.shape)
            
            for j in range(batch_size):
                step_logits = logits[episodes][j][:,:,:7]
                episode_actions = actions[episodes][j]
                predicted_labels = [0] * episode_actions.shape[0]
                for k, action in enumerate(episode_actions):
                    class_logits = step_logits[k,action]
                    prediction = torch.argmax(class_logits)
                    predicted_labels[action-1] = prediction
                episode_predicted_edge_scores = utils.matrix_to_edge_scores(
                        episodes[j],
                        predicted_labels,
                        torch.sigmoid(edge_logits[j]))
                episode_ground_truth_edge_scores = utils.matrix_to_edge_scores(
                        episodes[j],
                        episode_class_targets[episodes][j],
                        batch_edge_targets[j])
                all_predicted_edges.update(episode_predicted_edge_scores)
                all_ground_truth_edges.update(episode_ground_truth_edge_scores)
        
        '''
        pr, concave_pr, ap = evaluation.ap(
                edge_scores, edge_ground_truth, 0)
        concave_pr.append([0,1])
        '''
        
        pr, concave_pr, ap = evaluation.edge_ap(
                all_predicted_edges, all_ground_truth_edges)
        concave_pr.append([0,1])
        
        print('AP: %f'%ap)
        
        #edge_density = sum(edge_ground_truth) / len(edge_ground_truth)
        #print('Edge Density: %f'%edge_density)
        '''
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
        '''

with multi_environment:
    if args.test:
        model_checkpoint = './model_checkpoint_%04i.pt'%args.num_epochs
        model.load_state_dict(torch.load(model_checkpoint))
        model.eval()
        
        segmentation_checkpoint = (
                './segmentation_checkpoint_%04i.pt'%args.num_epochs)
        brick_classifier.load_state_dict(torch.load(segmentation_checkpoint))
        brick_classifier.eval()
        
        confidence_checkpoint = (
                './confidence_checkpoint_%04i.pt'%args.num_epochs)
        confidence_classifier.load_state_dict(torch.load(confidence_checkpoint))
        confidence_classifier.eval()
        
        edge_checkpoint = (
                './edge_checkpoint_%04i.pt'%args.num_epochs)
        edge_classifier.load_state_dict(torch.load(edge_checkpoint))
        edge_classifier.eval()
        
        #test_epoch(args.num_epochs, mark_selection=True, graph=True)
        
        class ModelWrapper():
            def __call__(self, observations, hidden_state):
                observations = [segs for im, segs in observations]
                x = bgt_utils.segments_to_tensor(observations).cuda()
                ep, inst, ch, h, w = x.shape
                x = model(x.view(ep*inst, ch, h, w))
                node_logits = brick_classifier(x)
                confidence_logits = confidence_classifier(x)
                #edge_logits = edge_classifier(x.view(ep, inst, -1))
                
                step_node_predictions = torch.argmax(
                        node_logits.view(ep, inst, -1), dim=-1)
                
                x = x.view(ep, inst, -1)
                if hidden_state is None:
                    hidden_state = (
                            torch.zeros_like(step_node_predictions).cpu(),
                            torch.zeros_like(x).cpu())
                
                label_accumulator, feature_accumulator = hidden_state
                
                confidence = torch.softmax(confidence_logits, dim=-1)[:,1]
                confidence = confidence.view(ep, inst)
                confidence = confidence * (step_node_predictions != 0)
                action = torch.argmax(confidence, dim=-1)
                
                if args.accumulator_mode == 'argmax':
                    label_accumulator[range(ep),action] = (
                            step_node_predictions[range(ep),action].cpu())
                    feature_accumulator[range(ep),action] = (
                            x[range(ep),action].cpu())
                elif args.accumulator_mode == 'all':
                    zero_prediction = step_node_predictions == 0
                    label_accumulator = (
                            label_accumulator * zero_prediction.cpu() +
                            (step_node_predictions * ~zero_prediction).cpu())
                    feature_accumulator = (
                            feature_accumulator *
                            zero_prediction.unsqueeze(-1).cpu() +
                            (x * ~zero_prediction.unsqueeze(-1)).cpu())
                
                edge_logits = edge_classifier(
                        feature_accumulator.cuda().view(ep, inst, -1))
                edge_matrix = torch.sigmoid(edge_logits)
                
                return (action.cpu(),
                        label_accumulator.cpu(),
                        edge_matrix.cpu(),
                        (label_accumulator.cpu(), feature_accumulator.cpu()))
        
        test_multi_env = MultiClass(
                args.num_processes,
                GraphEnv,
                [{'dataset' : 'random_stack',
                  'split' : 'test_mpd',
                  'viewpoint_control' : viewpoint_control,
                  'subset' : args.test_subset,
                  'rank' : i,
                  'size' : args.num_processes}
                 for i in range(args.num_processes)])
        
        with test_multi_env, torch.no_grad():
            m = ModelWrapper()
            step_ap = evaluation.dataset_node_and_edge_ap(
                    m, test_multi_env, dump_images=False)

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
                print('Saving main model checkpoint to: %s'%
                        model_checkpoint_path)
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
