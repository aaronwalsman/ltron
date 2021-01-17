import time
import math
import os
import json

import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy

import PIL.Image as Image

import tqdm

from gym.vector.async_vector_env import AsyncVectorEnv

import renderpy.masks as masks

from brick_gym.dataset.paths import get_dataset_info
from brick_gym.gym.brick_env import async_brick_env
from brick_gym.gym.standard_envs import segmentation_supervision_env
import brick_gym.torch.models.named_models as named_models
import brick_gym.torch.utils as utils
import brick_gym.visualization.image_generators as image_generators
from brick_gym.visualization.gym_dump import gym_dump

def train_label_confidence(
        num_epochs,
        mini_epochs_per_epoch,
        dataset,
        train_split = 'train',
        train_subset = None,
        test_split = 'test',
        test_subset = None,
        num_processes = 4,
        batch_size = 2,
        learning_rate = 3e-4,
        instance_label_loss_weight = 0.8,
        score_loss_weight = 0.2,
        score_ratio = 0.1,
        train_steps_per_epoch = 4096,
        test_frequency = 1,
        test_steps_per_epoch = 1024,
        model_output_channels = 32,
        checkpoint_frequency = 1,
        dump_train = False,
        dump_test = False,
        randomize_viewpoint = True,
        randomize_colors = True):
    
    step_clock = [0]
    log = SummaryWriter()
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['class_ids'].values()) + 1
    
    print('='*80)
    print('Building the model')
    model = named_models.named_graph_step_model(
            'nth_try',
            num_classes = num_classes).cuda()
    
    print('Building the optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print('Building the train environment')
    train_env = async_brick_env(
            num_processes,
            segmentation_supervision_env,
            dataset = dataset,
            split = train_split,
            randomize_viewpoint = randomize_viewpoint,
            randomize_viewpoint_frequency = 'step',
            randomize_colors = randomize_colors)
    
    print('Building the test environment')
    test_env = async_brick_env(
            num_processes,
            segmentation_supervision_env,
            dataset = dataset,
            split = test_split,
            randomize_viewpoint = randomize_viewpoint,
            randomize_viewpoint_frequency = 'reset',
            randomize_colors = False)
    
    for epoch in range(1, num_epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        train_label_confidence_epoch(
                epoch,
                step_clock,
                log,
                math.ceil(train_steps_per_epoch / num_processes),
                mini_epochs_per_epoch,
                batch_size,
                model,
                optimizer,
                train_env,
                instance_label_loss_weight,
                score_ratio,
                score_loss_weight,
                dataset_info,
                dump_train)
        
        if epoch % checkpoint_frequency == 0:
            print('-'*80)
            model_path = './model_%04i.pt'%epoch
            print('Saving model to: %s'%model_path)
            torch.save(model.state_dict(), model_path)
            
            optimizer_path = './optimizer_%04i.pt'%epoch
            print('Saving optimizer to: %s'%optimizer_path)
            torch.save(optimizer.state_dict(), optimizer_path)
        
        if epoch % test_frequency == 0:
            test_label_confidence_epoch(
                    epoch,
                    step_clock,
                    log,
                    math.ceil(test_steps_per_epoch / num_processes),
                    model,
                    test_env,
                    dump_test)
            
        
        print('-'*80)
        print('Elapsed: %.04f'%(time.time() - epoch_start))

def test_checkpoint(
        checkpoint,
        dataset,
        test_split = 'test',
        test_subset = None,
        num_processes = 4,
        test_steps = 4096,
        debug_dump = False,
        randomize_viewpoint = True):
    
    step_clock = [0]
    log = SummaryWriter()
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['class_ids'].values()) + 1
    
    print('='*80)
    print('Building the model')
    model = named_models.named_graph_step_model(
            'nth_try',
            node_classes = num_classes).cuda()
    model_state_dict = torch.load(checkpoint)
    model.load_state_dict(model_state_dict)
    
    print('Building the test environment')
    test_env = async_brick_env(
            num_processes,
            segmentation_supervision_env,
            dataset = dataset,
            split = test_split,
            randomize_viewpoint = randomize_viewpoint,
            randomize_viewpoint_frequency = 'reset',
            randomize_colors = False)
    
    test_label_confidence_epoch(
            0,
            step_clock,
            log,
            math.ceil(test_steps / num_processes),
            model,
            test_env,
            debug_dump)

def train_label_confidence_epoch(
        epoch,
        step_clock,
        log,
        steps,
        mini_epochs,
        batch_size,
        model,
        optimizer,
        train_env,
        instance_label_loss_weight,
        score_ratio,
        score_loss_weight,
        dataset_info,
        debug_dump):
    
    print('-'*80)
    print('Train')
    print('- '*40)
    print('Generating Data')
    
    if debug_dump:
        dump_directory = './dump_train_%i'%epoch
        if not os.path.exists(dump_directory):
            os.makedirs(dump_directory)
    
    observations = []
    step_observations = train_env.reset()
    
    for step in tqdm.tqdm(range(steps)):
        with torch.no_grad():
            # store observation
            observations.append(step_observations)
            
            # gym -> torch
            images = step_observations['color_render']
            x_im = utils.images_to_tensor(images).cuda()
            batch_size = x_im.shape[0]
            segmentations = step_observations['segmentation_render']
            x_seg = utils.segmentations_to_tensor(segmentations).cuda()
            
            # model forward pass
            #(node_logits,
            # action_logits,
            # segment_ids,
            # segment_weights) = model(x_im, x_seg)
            batch_graph, _, dense_scores, head_features = model(x_im, x_seg)
            
            # sample an action
            #hide_probs = torch.exp(batch_graph.hide_action.view(-1))
            hide_probs = batch_graph.score
            hide_logits = torch.log(hide_probs / (1. - hide_probs))
            actions = []
            for b in range(batch_size):
                start, end = batch_graph.ptr[b], batch_graph.ptr[b+1]
                # this is what we should be doing when hide_probs are not
                # repurposed scores
                #=======
                #action_distribution = Categorical(
                #        probs=hide_probs[start:end])
                #=======
                # this is temporary and it sucks, but we're doing it because
                # we're not training hide yet, and using the score as a hide
                # function
                action_distribution = Categorical(
                        logits = hide_logits[start:end])
                #=======
                action = action_distribution.sample()
                hide_segment = int(
                        batch_graph.segment_index[start+action].cpu())
                actions.append({'visibility':hide_segment})
            
            # debug
            if debug_dump:
                # dump observation
                gym_dump(step_observations,
                        train_env.single_observation_space,
                        os.path.join(dump_directory, 'observation_%06i'%step))
                
                # dump action info
                inverse_class_ids = dict(zip(
                        dataset_info['class_ids'].values(),
                        dataset_info['class_ids'].keys()))
                instance_labels = step_observations['instance_labels']
                action_brick_types = [
                        inverse_class_ids[instance_labels[i][index]]
                        for i, index in enumerate(hide_indices)]
                action_data = {
                    #'action_distribution' : action_prob.cpu().tolist(),
                    'segment_ids' : segment_ids.cpu().tolist(),
                    #'segment_weights' : segment_weights.cpu().tolist(),
                    'action_segment_indices' : hide_actions.tolist(),
                    'action_instance_indices' : hide_indices,
                    'action_brick_types' : action_brick_types
                }
                action_path = os.path.join(
                        dump_directory, 'action_%06i.json'%step)
                with open(action_path, 'w') as f:
                    json.dump(action_data, f, indent=2)
                
                # dump action mask
                num_images = x_im.shape[0]
                for i in range(num_images):
                    action_mask = segmentations[i] == int(hide_indices[i])
                    action_mask = action_mask.astype(numpy.uint8) * 255
                    dump_path = os.path.join(
                            dump_directory,
                            'action_mask_%06i_%02i.png'%(step, i))
                    Image.fromarray(action_mask).save(dump_path)
            
            step_observations, _, _, _ = train_env.step(actions)
    
    seq_x_im = torch.cat(tuple(
            utils.images_to_tensor(observation['color_render'])
            for observation in observations))
    seq_x_seg = torch.cat(tuple(
            utils.segmentations_to_tensor(observation['segmentation_render'])
            for observation in observations))
    y = torch.cat(tuple(
            torch.LongTensor(observation['instance_labels'])
            for observation in observations))
    
    running_node_loss = 0.
    running_confidence_loss = 0.
    total_correct_segments = 0
    total_correct_correct_segments = 0
    total_segments = 0
    
    dataset_size = seq_x_im.shape[0]
    
    for mini_epoch in range(1, mini_epochs+1):
        print('- '*40)
        print('Training Mini Epoch: %i'%mini_epoch)
        indices = torch.randperm(dataset_size)
        iterate = tqdm.tqdm(range(0, dataset_size, batch_size))
        for start in iterate:
            batch_indices = indices[start:start+batch_size]
            x_im = seq_x_im[batch_indices].cuda()
            x_seg = seq_x_seg[batch_indices].cuda()
            
            batch_graph, _, dense_scores, head_features = model(x_im, x_seg)
            
            y_batch = y[batch_indices].cuda()
            
            loss = 0.
            
            instance_label_target = torch.stack([
                    y[seg] for y,seg in zip(y_batch, x_seg)])
            instance_label_logits = head_features['instance_label']
            instance_label_loss = torch.nn.functional.cross_entropy(
                    instance_label_logits, instance_label_target)
            loss = loss + instance_label_loss * instance_label_loss_weight
            
            log.add_scalar(
                    'loss/instance_label', instance_label_loss, step_clock[0])
            
            '''
            prediction = torch.argmax(flat_node_logits, dim=-1)
            correct = (flat_y_batch == prediction) * flat_segment_weights
            flat_action_logits = action_logits.view(batch_size * num_segments)
            flat_action_prob = torch.sigmoid(flat_action_logits)
            confidence_loss = torch.nn.functional.binary_cross_entropy(
                    flat_action_prob, correct.float(), reduction='none')
            confidence_loss = confidence_loss * flat_segment_weights
            confidence_loss = (
                    confidence_loss * correct * score_ratio +
                    confidence_loss * ~correct * 1.0)
            confidence_loss = (
                    torch.sum(confidence_loss) / normalizer)
            loss = loss + confidence_loss * score_loss_weight
            '''
            instance_label_prediction = torch.argmax(
                    instance_label_logits, dim=1)
            correct = instance_label_prediction == instance_label_target
            score_loss = torch.nn.functional.binary_cross_entropy(
                    dense_scores,
                    (correct & (x_seg != 0)).unsqueeze(1).float(),
                    reduction='none')
            score_loss = torch.mean(
                    score_loss * correct * score_ratio + score_loss * ~correct)
            loss = loss + score_loss * score_loss_weight
            
            log.add_scalar('loss/score', score_loss, step_clock[0])
            log.add_scalar('loss/total', loss, step_clock[0])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            step_clock[0] += 1
    
    #print('- '*40)
    #print('Node Accuracy: %.04f'%(total_correct_segments/total_segments))
    #print('Confidence Accuracy: %.04f'%(
    #        total_correct_correct_segments/total_segments))

def test_label_confidence_epoch(
        epoch,
        step_clock,
        log,
        steps,
        model,
        test_env,
        debug_dump=False):
    
    print('-'*80)
    print('Test')
    
    step_observations = test_env.reset()
    
    total_correct_segments = 0
    total_correct_correct_segments = 0
    total_segments = 0
    add_to_graph_correct = 0
    add_to_graph_normalizer = 1e-5
    #max_is_correct_segments = 0
    #max_is_correct_normalizer = 0
    for step in tqdm.tqdm(range(steps)):
        with torch.no_grad():
            
            images = step_observations['color_render']
            x_im = utils.images_to_tensor(images).cuda()
            batch_size = x_im.shape[0]
            segmentations = step_observations['segmentation_render']
            x_seg = utils.segmentations_to_tensor(segmentations).cuda()
            
            batch_graph, _, dense_scores, head_features = model(x_im, x_seg)
            
            y_batch = torch.LongTensor(
                    step_observations['instance_labels']).cuda()
            
            instance_label_logits = batch_graph.instance_label
            segment_index = batch_graph.segment_index
            ptr = batch_graph.ptr
            instance_label_targets = torch.cat([
                    y[segment_index[start:end]]
                    for y, start, end in zip(y_batch, ptr[:-1], ptr[1:])])
            prediction = torch.argmax(instance_label_logits, dim=1)
            correct = instance_label_targets == prediction
            total_correct_segments += float(torch.sum(correct).cpu())
            total_segments += correct.shape[0]
            
            #action_prob = torch.exp(action_logits) * segment_weights
            #max_action = torch.argmax(action_prob, dim=-1)
            #max_correct = correct[range(batch_size), max_action]
            #max_is_correct_segments += int(torch.sum(max_correct))
            #max_is_correct_normalizer += batch_size
            # this is bad, but we're not training the hide_actions yet
                
            '''
            action_prob = torch.sigmoid(action_logits) * segment_weights
            max_action = torch.argmax(action_prob, dim=-1)
            add_to_graph = action_prob > 0.5
            add_to_graph_correct += torch.sum(correct * add_to_graph)
            add_to_graph_normalizer += torch.sum(add_to_graph)
            '''
            
            if debug_dump:
                for i in range(batch_size):
                    Image.fromarray(images[i]).save(
                            'image_%i_%i_%i.png'%(epoch, i, step))
                    max_index = segment_ids[i,max_action[i]]
                    max_highlight = segmentations[i] == int(max_index)
                    Image.fromarray(max_highlight.astype(numpy.uint8)*255).save(
                            'highlight_%i_%i_%i.png'%(epoch, i, step))
                    
                    action_weights = (
                            torch.sigmoid(action_logits[i]) *
                            segment_weights[i])
                    confidence_image = image_generators.segment_weight_image(
                            segmentations[i],
                            action_weights.cpu().numpy(),
                            segment_ids[i].cpu().numpy())
                    Image.fromarray(confidence_image).save(
                            'segment_confidence_%i_%i_%i.png'%(epoch, i, step))
                    correct_image = image_generators.segment_weight_image(
                            segmentations[i],
                            correct[i].cpu().numpy(),
                            segment_ids[i].cpu().numpy())
                    Image.fromarray(correct_image).save(
                            'correct_%i_%i_%i.png'%(epoch, i, step))
                    
                    metadata = {
                        'max_action' : int(max_action[i]),
                        'max_label' : int(y[i,max_action[i]]),
                        'max_prediction' : int(prediction[i,max_action[i]]),
                        'all_labels' : y[i].cpu().tolist(),
                        'all_predictions' : prediction[i].cpu().tolist()
                    }
                    with open('details_%i_%i_%i.json'%(epoch,i,step), 'w') as f:
                        json.dump(metadata, f)
            
            '''
            confidence_prediction = action_logits > 0.
            correct_correct = (
                    (confidence_prediction == correct) * segment_weights)
            total_correct_correct_segments += float(
                    torch.sum(correct_correct).cpu())
            '''
            '''
            #action_distribution = Categorical(logits=action_logits)
            #hide_actions = action_distribution.sample().cpu()
            hide_actions = max_action
            hide_indices = [segment_ids[i,action]
                    for i, action in enumerate(hide_actions)]
            actions = [{'visibility':int(action)} for action in hide_indices]
            '''
            
            hide_probs = batch_graph.score
            hide_logits = torch.log(hide_probs / (1. - hide_probs))
            actions = []
            for b in range(batch_size):
                start, end = batch_graph.ptr[b], batch_graph.ptr[b+1]
                # this is what we should be doing when hide_probs are not
                # repurposed scores
                #=======
                #action_distribution = Categorical(
                #        probs=hide_probs[start:end])
                #=======
                # this is temporary and it sucks, but we're doing it because
                # we're not training hide yet, and using the score as a hide
                # function
                action_distribution = Categorical(
                        logits = hide_logits[start:end])
                #=======
                action = action_distribution.sample()
                hide_segment = int(
                        batch_graph.segment_index[start+action].cpu())
                actions.append({'visibility':hide_segment})
            
            step_observations, _, terminal, _ = test_env.step(actions)
    
    print('- '*40)
    node_accuracy = total_correct_segments/total_segments
    #confidence_accuracy = total_correct_correct_segments/total_segments
    #top_confidence_accuracy = (
    #        add_to_graph_correct / add_to_graph_normalizer)
            #max_is_correct_segments / max_is_correct_normalizer)
    
    log.add_scalar('test_accuracy/node_labels', node_accuracy, step_clock[0])
    #log.add_scalar('test_accuracy/confidence_accuracy',
    #        confidence_accuracy, step_clock[0])
    #log.add_scalar('test_accuracy/confident_node_accuracy',
    #        top_confidence_accuracy, step_clock[0])
