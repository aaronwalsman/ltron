import time
import math
import os
import json

import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import binary_cross_entropy

import numpy

import PIL.Image as Image

import tqdm

import torch_geometric.utils as tg_utils

import renderpy.masks as masks
from renderpy.json_numpy import NumpyEncoder

import brick_gym.evaluation as evaluation
from brick_gym.dataset.paths import get_dataset_info
from brick_gym.gym.brick_env import async_brick_env
from brick_gym.gym.standard_envs import graph_supervision_env
import brick_gym.visualization.image_generators as image_generators

from brick_gym.torch.brick_geometric import (
        BrickList, BrickGraph, BrickGraphBatch)
from brick_gym.torch.gym_tensor import (
        gym_space_to_tensors, gym_space_list_to_tensors, graph_to_gym_space)
from brick_gym.torch.gym_log import gym_log
import brick_gym.torch.models.named_models as named_models
from brick_gym.torch.train.loss import dense_score_loss, cross_product_loss

edge_threshold = 0.05

def train_label_confidence(
        # load checkpoints
        step_checkpoint = None,
        edge_checkpoint = None,
        optimizer_checkpoint = None,
        
        # general settings
        num_epochs = 9999,
        mini_epochs_per_epoch = 1,
        mini_epoch_sequences = 2048,
        mini_epoch_sequence_length = 2,
        
        # dasaset settings
        dataset = 'random_stack',
        num_processes = 8,
        train_split = 'train',
        train_subset = None,
        test_split = 'test',
        test_subset = None,
        randomize_viewpoint = True,
        randomize_colors = True,
        
        # train settings
        train_steps_per_epoch = 4096,
        batch_size = 6,
        learning_rate = 3e-4,
        instance_label_loss_weight = 0.8,
        instance_label_background_weight = 0.05,
        score_loss_weight = 0.2,
        score_background_weight = 0.05,
        score_ratio = 0.1,
        matching_loss_weight = 1.0,
        edge_loss_weight = 1.0,
        
        # test settings
        test_frequency = 1,
        test_steps_per_epoch = 1024,
        
        # checkpoint settings
        checkpoint_frequency = 1,
        
        # logging settings
        log_train = 0,
        log_test = 0):
    
    # logging
    step_clock = [0]
    log = SummaryWriter()
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['class_ids'].values()) + 1
    
    # background class weight
    instance_label_class_weight = torch.ones(num_classes)
    instance_label_class_weight[0] = instance_label_background_weight
    instance_label_class_weight = instance_label_class_weight.cuda()
    
    print('='*80)
    print('Building the step model')
    step_model = named_models.named_graph_step_model(
            'nth_try',
            num_classes = num_classes).cuda()
    if step_checkpoint is not None:
        print('Loading step model checkpoint from:')
        print(step_checkpoint)
        step_model.load_state_dict(torch.load(step_checkpoint))
    
    print('-'*80)
    print('Building the edge model')
    edge_model = named_models.named_edge_model(
            'subtract',
            input_dim=256).cuda()
    if edge_checkpoint is not None:
        print('Loading edge model checkpoint from:')
        print(edge_checkpoint)
        edge_model.load_state_dict(torch.load(edge_checkpoint))
    
    print('-'*80)
    print('Building the optimizer')
    optimizer = torch.optim.Adam(
            list(step_model.parameters()) + list(edge_model.parameters()),
            lr=learning_rate)
    if optimizer_checkpoint is not None:
        print('Loading optimizer checkpoint from:')
        print(optimizer_checkpoint)
        optimizer.load_state_dict(torch.load(optimizer_checkpoint))
    
    print('='*80)
    print('Building the train environment')
    train_env = async_brick_env(
            num_processes,
            graph_supervision_env,
            dataset = dataset,
            split = train_split,
            randomize_viewpoint = randomize_viewpoint,
            randomize_viewpoint_frequency = 'reset',
            randomize_colors = randomize_colors)
    
    '''
    print('-'*80)
    print('Building the test environment')
    test_env = async_brick_env(
            num_processes,
            graph_supervision_env,
            dataset = dataset,
            split = test_split,
            randomize_viewpoint = randomize_viewpoint,
            randomize_viewpoint_frequency = 'reset',
            randomize_colors = False)
    '''
    
    for epoch in range(1, num_epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        train_label_confidence_epoch(
                epoch,
                step_clock,
                log,
                math.ceil(train_steps_per_epoch / num_processes),
                #train_steps_per_epoch,
                mini_epochs_per_epoch,
                mini_epoch_sequences,
                mini_epoch_sequence_length,
                batch_size,
                step_model,
                edge_model,
                optimizer,
                train_env,
                instance_label_loss_weight,
                instance_label_class_weight,
                score_loss_weight,
                score_background_weight,
                score_ratio,
                matching_loss_weight,
                edge_loss_weight,
                dataset_info,
                log_train)
        
        if (checkpoint_frequency is not None and
                epoch % checkpoint_frequency) == 0:
            checkpoint_directory = os.path.join(
                    './checkpoint', os.path.split(log.log_dir)[-1])
            if not os.path.exists(checkpoint_directory):
                os.makedirs(checkpoint_directory)
            
            print('-'*80)
            step_model_path = os.path.join(
                    checkpoint_directory, 'step_model_%04i.pt'%epoch)
            print('Saving step_model to: %s'%step_model_path)
            torch.save(step_model.state_dict(), step_model_path)
            
            edge_model_path = os.path.join(
                    checkpoint_directory, 'edge_model_%04i.pt'%epoch)
            print('Saving edge_model to: %s'%edge_model_path)
            torch.save(edge_model.state_dict(), edge_model_path)
            
            optimizer_path = os.path.join(
                    checkpoint_directory, 'optimizer_%04i.pt'%epoch)
            print('Saving optimizer to: %s'%optimizer_path)
            torch.save(optimizer.state_dict(), optimizer_path)
        
        if test_frequency is not None and epoch % test_frequency == 0:
            test_label_confidence_epoch(
                    epoch,
                    step_clock,
                    log,
                    math.ceil(test_steps_per_epoch / num_processes),
                    step_model,
                    edge_model,
                    test_env,
                    log_test)
        
        print('-'*80)
        print('Elapsed: %.04f'%(time.time() - epoch_start))

def test_checkpoint(
        step_checkpoint,
        edge_checkpoint,
        dataset,
        test_split = 'test',
        test_subset = None,
        num_processes = 4,
        test_steps = 4096,
        log_debug = False,
        randomize_viewpoint = True):
    
    step_clock = [0]
    log = SummaryWriter()
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['class_ids'].values()) + 1
    
    print('='*80)
    print('Building the step_model')
    step_model = named_models.named_graph_step_model(
            'nth_try',
            node_classes = num_classes).cuda()
    model_state_dict = torch.load(checkpoint)
    step_model.load_state_dict(model_state_dict)
    
    print('='*80)
    print('Building the edge model')
    edge_model = named_models.named_edge_model(
            'default',
            input_dim=256,
            output_dim=2).cuda()
    
    print('Building the test environment')
    test_env = async_brick_env(
            num_processes,
            graph_supervision_env,
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
            step_model,
            edge_model,
            test_env,
            log_debug)

def train_label_confidence_epoch(
        epoch,
        step_clock,
        log,
        steps,
        mini_epochs,
        mini_epoch_sequences,
        mini_epoch_sequence_length,
        batch_size,
        step_model,
        edge_model,
        optimizer,
        train_env,
        instance_label_loss_weight,
        instance_label_class_weight,
        score_loss_weight,
        score_background_weight,
        score_ratio,
        matching_loss_weight,
        edge_loss_weight,
        dataset_info,
        log_debug):
    
    print('-'*80)
    print('Train')
    print('- '*40)
    print('Generating Data')
    
    seq_terminal = []
    seq_observations = []
    seq_graph_state = []
    seq_rewards = []
    
    device = torch.device('cuda:0')
    
    step_observations = train_env.reset()
    step_terminal = numpy.ones(train_env.num_envs, dtype=numpy.bool)
    step_rewards = numpy.zeros(train_env.num_envs)
    graph_states = [None] * train_env.num_envs
    for step in tqdm.tqdm(range(steps)):
        with torch.no_grad():
            # store observation
            seq_terminal.append(step_terminal)
            seq_observations.append(step_observations)
            seq_graph_state.append(graph_states)
            
            # gym -> torch
            step_tensors = gym_space_to_tensors(
                    step_observations,
                    train_env.single_observation_space,
                    device)
            
            # step model forward pass
            step_brick_lists, _, dense_scores, head_features = step_model(
                    step_tensors['color_render'],
                    step_tensors['segmentation_render'])
            
            # build new graph state for all terminal sequences
            # (do this here so we can use the brick_feature_spec from the model)
            for i, terminal in enumerate(step_terminal):
                if terminal:
                    graph_states[i] = BrickGraph(
                            BrickList(step_brick_lists[i].brick_feature_spec()),
                            edge_attr_channels=1).cuda()
            
            # store the graph_state before it gets updated
            # (we want the graph state that was input to this step)
            seq_graph_state.append(graph_states)
            
            # upate the graph state using the edge model
            graph_states, step_step_logits, step_state_logits = edge_model(
                    step_brick_lists, graph_states)
            
            # sample an action
            hide_logits = [sbl.hide_action.view(-1) for sbl in step_brick_lists]
            hide_distributions = []
            bad_distributions = []
            for i, logits in enumerate(hide_logits):
                try:
                    distribution = Categorical(
                            logits=logits, validate_args=True)
                except ValueError:
                    bad_distrbutions.append(i)
                    distribution = Categorical(probs=torch.ones(1).cuda())
                hide_distributions.append(distribution)
            
            if len(bad_distributions):
                print('BAD DISTRIBUTIONS, REVERTING TO 0')
                print('STEP: %i, GLOBAL_STEP: %i'%(step, step_clock[0]))
                print('BAD INDICES:', bad_distributions)
                space = train_env.single_observation_space
                log_train_rollout_step(
                        step_clock, log, step_observations, space, [])
            
            segment_samples = [dist.sample() for dist in hide_distributions]
            instance_samples = [brick_list.segment_id[sample]
                    for brick_list, sample
                    in zip(step_brick_lists, segment_samples)]
            actions = []
            for i, graph_state in zip(instance_samples, graph_states):
                graph_data = graph_to_gym_space(
                        graph_state,
                        train_env.single_action_space['graph_task'])
                actions.append({
                        'visibility':int(i.cpu()),
                        'graph_task':graph_data})
            
            if step < log_debug:
                space = train_env.single_observation_space
                log_train_rollout_step(
                        step_clock,
                        log,
                        step_observations,
                        space,
                        actions)
            
            (step_observations,
             step_rewards,
             step_terminal,
             step_info) = train_env.step(actions)
            
            log.add_scalar(
                    'train_rollout/reward',
                    sum(step_rewards)/len(step_rewards),
                    step_clock[0])
            
            all_edge_ap = [
                    info['graph_task']['edge_ap'] for info in step_info]
            all_instance_ap = [
                    info['graph_task']['instance_ap'] for info in step_info]
            
            log.add_scalar(
                    'train_rollout/step_edge_ap',
                    sum(all_edge_ap)/len(all_edge_ap),
                    step_clock[0])
            log.add_scalar(
                    'train_rollout/step_instance_ap',
                    sum(all_instance_ap)/len(all_instance_ap),
                    step_clock[0])
            
            num_terminal = sum(step_terminal)
            if num_terminal:
                '''
                terminal_info = [
                        info for t, info in zip(step_terminal, step_info) if t]
                sum_terminal_edge_ap = sum(
                        info['task']['edge_ap'] for info in terminal_info)
                sum_terminal_instance_ap = sum(
                        info['task']['instance_ap'] for info in terminal_info)
                '''
                sum_terminal_edge_ap = sum(
                        ap * t for ap, t in zip(all_edge_ap, step_terminal))
                sum_terminal_instance_ap = sum(
                        ap * t for ap, t in zip(all_instance_ap, step_terminal))
                log.add_scalar(
                        'train_rollout/terminal_edge_ap',
                        sum_terminal_edge_ap/num_terminal, step_clock[0])
                log.add_scalar(
                        'train_rollout/terminal_instance_ap',
                        sum_terminal_instance_ap/num_terminal, step_clock[0])
            
            seq_rewards.append(step_rewards)
            
            step_clock[0] += 1
    
    # when joining these into one long list, make sure sequences are preserved
    seq_tensors = gym_space_list_to_tensors(
            seq_observations, train_env.single_observation_space)
    seq_terminal = numpy.stack(seq_terminal, axis=1).reshape(-1)
    seq_graph_state = BrickGraphBatch([
            seq_graph_state[i][j]
            for j in range(train_env.num_envs)
            for i in range(steps)])
    
    running_node_loss = 0.
    running_confidence_loss = 0.
    total_correct_segments = 0
    total_correct_correct_segments = 0
    total_segments = 0
    
    dataset_size = seq_tensors['color_render'].shape[0]
    tlast = 0
    for mini_epoch in range(1, mini_epochs+1):
        print('- '*40)
        print('Training Mini Epoch: %i'%mini_epoch)
        
        iterate = tqdm.tqdm(range(mini_epoch_sequences//batch_size))
        for seq_id in iterate:
            start_indices = torch.randint(dataset_size, (batch_size,))
            step_terminal = [True for _ in range(batch_size)]
            graph_states = [None for _ in range(batch_size)]
            seq_loss = 0.
            
            ####################################
            #torch.cuda.synchronize()
            #t0 = tlast
            #tlast = time.time()
            #print('ploop', tlast - t0)
            
            for step in range(mini_epoch_sequence_length):
                step_indices = (start_indices + step) % dataset_size
                
                # gym -> torch
                x_im = seq_tensors['color_render'][step_indices].cuda()
                x_seg = seq_tensors['segmentation_render'][step_indices].cuda()
                y_graph = seq_tensors['graph_label'][step_indices].cuda()
                
                #################################
                #torch.cuda.synchronize()
                #t2 = time.time()
                
                # step forward pass
                step_brick_lists, _, dense_scores, head_features = step_model(
                        x_im, x_seg)
                
                ################################
                #torch.cuda.synchronize()
                #t3 = time.time()
                #print('fwd', t3 - t2)
                
                # select graph state from memory for all terminal sequences
                # TODO: Is there more we need to do here to make sure gradients
                #       get clipped here?
                for i, terminal in enumerate(step_terminal):
                    if terminal:
                        graph_states[i] = seq_graph_state[
                                step_indices[i]].cuda().detach()
                
                # update step terminal
                step_terminal = seq_terminal[step_indices]
                
                ################################
                #torch.cuda.synchronize()
                #t4 = time.time()
                #print('term', t4 - t3)
                
                # upate the graph state using the edge model
                (new_graph_states,
                 step_step_logits,
                 step_state_logits) = edge_model(
                        step_brick_lists, graph_states)
                
                ################################
                #torch.cuda.synchronize()
                #t5 = time.time()
                #print('edge', t5 - t4)
                
                step_loss = 0.
                
                # instance_label
                y_instance_label = [
                        graph['instance_label'][:,0] for graph in y_graph]
                dense_instance_label_target = torch.stack([
                        y[seg] for y,seg in zip(y_instance_label, x_seg)])
                instance_label_logits = head_features['instance_label']
                instance_label_loss = torch.nn.functional.cross_entropy(
                        instance_label_logits,
                        dense_instance_label_target,
                        weight=instance_label_class_weight)
                step_loss = (step_loss +
                        instance_label_loss * instance_label_loss_weight)
                log.add_scalar(
                        'loss/instance_label',
                        instance_label_loss, step_clock[0])
                
                foreground = x_seg != 0
                foreground_total = torch.sum(foreground)
                if foreground_total:
                    # score
                    instance_label_prediction = torch.argmax(
                            instance_label_logits, dim=1)
                    correct = (
                            instance_label_prediction ==
                            dense_instance_label_target)
                    score_loss = dense_score_loss(
                            dense_scores, correct, foreground)
                    step_loss = step_loss + score_loss * score_loss_weight
                    log.add_scalar('loss/score', score_loss, step_clock[0])
                    
                    #dense_instance_label_accuracy = (
                    log.add_scalar('train_accuracy/dense_instance_label',
                            float(torch.sum(correct)) /
                            float(torch.numel(correct)),
                            step_clock[0])
                
                    # visibility
                    dense_visibility = head_features['hide_action']
                    '''
                    visibility_loss = binary_cross_entropy(
                            torch.sigmoid(dense_visibility),
                            correct.unsqueeze(1).float(),
                            #(correct & (x_seg != 0)).unsqueeze(1).float(),
                            reduction='none')
                    #visibility_loss = torch.mean(
                    #        visibility_loss * correct * score_ratio +
                    #        visibility_loss * ~correct)
                    visibility_loss = torch.sum(
                            visibility_loss * (x_seg != 0)) / foreground_total
                    '''
                    visibility_loss = dense_score_loss(
                            torch.sigmoid(dense_visibility),
                            correct,
                            foreground)
                    step_loss = step_loss + visibility_loss * score_loss_weight
                    log.add_scalar(
                            'loss/visibility', visibility_loss, step_clock[0])
                
                instance_correct = 0.
                total_instances = 0.
                for brick_list, target_graph in zip(step_brick_lists, y_graph):
                    instance_label_target = (
                            target_graph.instance_label[
                                brick_list.segment_id[:,0]])[:,0]
                    instance_label_pred = torch.argmax(
                            brick_list.instance_label, dim=-1)
                    instance_correct += float(torch.sum(
                            instance_label_target == instance_label_pred).cpu())
                    total_instances += instance_label_pred.shape[0]
                log.add_scalar('train_accuracy/step_instance_label',
                        instance_correct / total_instances,
                        step_clock[0])
                
                ################################
                #torch.cuda.synchronize()
                #t6 = time.time()
                #print('label/score/vis loss', t6 - t5)
                
                # edges and matching
                matching_loss = 0.
                edge_loss = 0.
                normalizer = 0.
                step_step_matching_correct = 0.
                step_step_edge_correct = 0.
                step_step_normalizer = 0.
                step_state_matching_correct = 0.
                step_state_edge_correct = 0.
                step_state_normalizer = 0.
                for (step_step,
                     step_state,
                     brick_list,
                     graph_state,
                     y) in zip(
                        step_step_logits,
                        step_state_logits,
                        step_brick_lists,
                        graph_states,
                        y_graph):
                    
                    step_step = torch.sigmoid(step_step)
                    step_state = torch.sigmoid(step_state)
                    
                    '''
                    normalizer += brick_list.num_nodes**2
                    normalizer += (
                            brick_list.num_nodes * graph_state.num_nodes) * 2
                    '''
                    step_step_num = brick_list.num_nodes**2
                    step_step_normalizer += step_step_num
                    step_state_num = (
                            brick_list.num_nodes * graph_state.num_nodes) * 2
                    step_state_normalizer += step_state_num
                    normalizer += step_step_num + step_state_num
                    
                    # matching loss
                    step_step_matching_target = torch.eye(
                            brick_list.num_nodes, dtype=torch.bool).to(device)
                    '''
                    step_step_matching_loss = binary_cross_entropy(
                            step_step[:,:,0], step_step_matching_target,
                            reduction = 'sum')
                    '''
                    step_step_matching_loss = cross_product_loss(
                            step_step[:,:,0], step_step_matching_target)
                    step_step_matching_pred = step_step[:,:,0] > 0.5
                    step_step_matching_correct += torch.sum(
                            step_step_matching_pred ==
                            step_step_matching_target)
                    
                    step_state_matching_target = (
                            brick_list.segment_id ==
                            graph_state.segment_id.t())
                    '''
                    step_state_matching_loss = binary_cross_entropy(
                            step_state[:,:,0], step_state_matching_target,
                            reduction = 'sum') * 2
                    '''
                    step_state_matching_loss = cross_product_loss(
                            step_state[:,:,0], step_state_matching_target) * 2
                    step_state_matching_pred = step_state[:,:,0] > 0.5
                    step_state_matching_correct += torch.sum(
                            step_state_matching_pred ==
                            step_state_matching_target) * 2
                    
                    matching_loss = (matching_loss +
                            step_step_matching_loss * step_step_num +
                            step_state_matching_loss * step_state_num)
                    
                    # edge loss
                    full_edge_target = y.edge_matrix(bidirectionalize=True).to(
                            torch.bool)
                    
                    step_lookup = brick_list.segment_id[:,0]
                    step_step_edge_target = full_edge_target[
                            step_lookup][:,step_lookup]
                    '''
                    step_step_edge_loss = binary_cross_entropy(
                            step_step[:,:,1], step_step_edge_target,
                            reduction = 'sum')
                    '''
                    step_step_edge_loss = cross_product_loss(
                            step_step[:,:,1], step_step_edge_target)
                    step_step_edge_pred = step_step[:,:,1] > 0.5
                    step_step_edge_correct += torch.sum(
                            step_step_edge_pred ==
                            step_step_edge_target)
                    
                    state_lookup = graph_state.segment_id[:,0]
                    step_state_edge_target = full_edge_target[
                            step_lookup][:,state_lookup]
                    '''
                    step_state_edge_loss = binary_cross_entropy(
                            step_state[:,:,1], step_state_edge_target,
                            reduction = 'sum')
                    '''
                    step_state_edge_loss = cross_product_loss(
                            step_state[:,:,1], step_state_edge_target) * 2
                    step_state_edge_pred = step_state[:,:,1] > 0.5
                    step_state_edge_correct += torch.sum(
                            step_state_edge_pred ==
                            step_state_edge_target) * 2
                    
                    edge_loss = (edge_loss +
                            step_step_edge_loss * step_step_num +
                            step_state_edge_loss * step_state_num)
                
                matching_loss = matching_loss / normalizer
                step_loss = step_loss + matching_loss * matching_loss_weight
                log.add_scalar('loss/matching', matching_loss, step_clock[0])
                
                edge_loss = edge_loss / normalizer
                step_loss = step_loss + edge_loss * edge_loss_weight
                log.add_scalar('loss/edge', edge_loss, step_clock[0])
                
                step_step_match_acc = (
                        step_step_matching_correct / step_step_normalizer)
                log.add_scalar('train_accuracy/step_step_match',
                        step_step_match_acc, step_clock[0])
                step_step_edge_acc = (
                        step_step_edge_correct / step_step_normalizer)
                log.add_scalar('train_accuracy/step_step_edge',
                        step_step_edge_acc, step_clock[0])
                if step_state_normalizer:
                    step_state_match_acc = (
                            step_state_matching_correct / step_state_normalizer)
                    log.add_scalar('train_accuracy/step_state_match',
                            step_state_match_acc, step_clock[0])
                    step_state_edge_acc = (
                            step_state_edge_correct / step_state_normalizer)
                    log.add_scalar('train_accuracy/step_state_edge',
                            step_state_edge_acc, step_clock[0])
                
                log.add_scalar('loss/total', step_loss, step_clock[0])
                
                seq_loss = seq_loss + step_loss
                
                ################################
                #torch.cuda.synchronize()
                #t7 = time.time()
                #print('edge', t7 - t6)
                
                if seq_id < log_debug:
                    log_train_loss_step(
                            # log
                            step_clock,
                            log,
                            # input
                            x_im.cpu().numpy(),
                            x_seg.cpu().numpy(),
                            # predictions
                            instance_label_logits.cpu().detach(),
                            dense_scores.cpu().detach(),
                            step_brick_lists.cpu().detach(),
                            [l.cpu().detach() for l in step_step_logits],
                            [l.cpu().detach() for l in step_state_logits],
                            # ground truth
                            y_graph.cpu(),
                            correct)
                
                graph_states = new_graph_states
                step_clock[0] += 1
            
            ################################
            #torch.cuda.synchronize()
            #t8 = time.time()
            
            seq_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            ################################
            #torch.cuda.synchronize()
            #t9 = time.time()
            #print('backward, step', t9-t8)
    
    #print('- '*40)
    #print('Node Accuracy: %.04f'%(total_correct_segments/total_segments))
    #print('Confidence Accuracy: %.04f'%(
    #        total_correct_correct_segments/total_segments))

def log_train_rollout_step(
        step_clock,
        log,
        step_observations,
        space,
        actions):
    log_step_observations('train', step_clock, log, step_observations, space)
    log.add_text('train/actions',
            json.dumps(actions, cls=NumpyEncoder, indent=2))

def log_train_loss_step(
        # log
        step_clock,
        log,
        # input
        images,
        segmentation,
        # predictions
        dense_label_logits,
        dense_scores,
        pred_brick_lists,
        pred_step_step,
        pred_step_state,
        # ground truth
        true_graph,
        score_targets):
    
    #input
    log.add_image('train/train_images', images, step_clock[0],
            dataformats='NCHW')
    segmentation_mask = masks.color_index_to_byte(segmentation)
    log.add_image('train/train_segmentation', segmentation_mask, step_clock[0],
            dataformats='NHWC')
    
    # prediction
    dense_label_prediction = torch.argmax(dense_label_logits, dim=1).numpy()
    dense_label_mask = masks.color_index_to_byte(dense_label_prediction)
    log.add_image(
            'train/pred_dense_label_mask',
            dense_label_mask,
            step_clock[0],
            dataformats='NHWC')
    
    log.add_image(
            'train/pred_dense_scores',
            dense_scores,
            step_clock[0],
            dataformats='NCHW')
    
    true_class_label_images = []
    pred_class_label_images = []
    true_instance_label_lookups = []
    pred_instance_label_lookups = []
    for i in range(len(true_graph)):
        true_instance_label_lookup = true_graph[i].instance_label.numpy()[:,0]
        true_instance_label_lookups.append(true_instance_label_lookup)
        class_label_segmentation = true_instance_label_lookup[segmentation[i]]
        class_label_image = masks.color_index_to_byte(class_label_segmentation)
        true_class_label_images.append(class_label_image)
        
        pred_instance_label_lookup = numpy.zeros(
                true_instance_label_lookup.shape, dtype=numpy.long)
        pred_lookup_partial = (
                torch.argmax(pred_brick_lists[i].instance_label, dim=-1))
        pred_instance_label_lookup[pred_brick_lists[i].segment_id[:,0]] = (
                pred_lookup_partial.numpy())
        pred_instance_label_lookups.append(pred_instance_label_lookup)
        class_label_segmentation = pred_instance_label_lookup[segmentation[i]]
        class_label_image = masks.color_index_to_byte(class_label_segmentation)
        pred_class_label_images.append(class_label_image)
    
    pred_class_label_images = numpy.stack(pred_class_label_images)
    log.add_image(
            'train/pred_instance_label_mask',
            pred_class_label_images,
            step_clock[0],
            dataformats='NHWC')
    
    max_size = max(step_step.shape[0] for step_step in pred_step_step)
    step_step_match_image = torch.zeros(
            len(pred_step_step), 1, max_size, max_size)
    step_step_edge_image = torch.zeros(
            len(pred_step_step), 1, max_size, max_size)
    for i, step_step in enumerate(pred_step_step):
        size = step_step.shape[0]
        step_step_match_image[i, 0, :size, :size] = step_step[:,:,0]
        step_step_edge_image[i, 0, :size, :size] = step_step[:,:,1]
    log.add_image(
            'train/pred_step_step_match_score',
            torch.sigmoid(step_step_match_image),
            step_clock[0],
            dataformats='NCHW')
    log.add_image(
            'train/pred_step_step_edge_score',
            torch.sigmoid(step_step_edge_image),
            step_clock[0],
            dataformats='NCHW')
    
    max_h = max(step_state.shape[0] for step_state in pred_step_state)
    max_h = max(1, max_h)
    max_w = max(step_state.shape[1] for step_state in pred_step_state)
    max_w = max(1, max_w)
    step_state_match_image = torch.zeros(
            len(pred_step_state), 1, max_h, max_w)
    step_state_edge_image = torch.zeros(
            len(pred_step_state), 1, max_h, max_w)
    for i, step_state in enumerate(pred_step_state):
        h,w = step_state.shape[:2]
        step_state_match_image[i, 0, :h, :w] = step_state[:,:,0]
        step_state_edge_image[i, 0, :h, :w] = step_state[:,:,1]
    log.add_image(
            'train/pred_step_state_match_score',
            torch.sigmoid(step_state_match_image),
            step_clock[0],
            dataformats='NCHW')
    log.add_image(
            'train/pred_step_state_edge_score',
            torch.sigmoid(step_state_edge_image),
            step_clock[0],
            dataformats='NCHW')
    
    
    log.add_text('train/pred_instance_labels',
            json.dumps(pred_instance_label_lookups,
                    cls=NumpyEncoder, indent=2),
            step_clock[0])
    
    # ground truth
    true_class_label_images = numpy.stack(true_class_label_images)
    log.add_image(
            'train/true_instance_label_mask',
            true_class_label_images,
            step_clock[0],
            dataformats='NHWC')
    
    log.add_image('train/true_score_labels',
            score_targets.unsqueeze(1),
            step_clock[0],
            dataformats='NCHW')
    
    log.add_text('train/true_instance_labels',
            json.dumps(true_instance_label_lookups,
                    cls=NumpyEncoder, indent=2),
            step_clock[0])

def log_step_observations(split, step_clock, log, step_observations, space):
    
    label = '%s/observations'%split
    gym_log(label, step_observations, space, log, step_clock[0])
    
    
    '''
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
    '''

def test_label_confidence_epoch(
        epoch,
        step_clock,
        log,
        steps,
        step_model,
        edge_model,
        test_env,
        debug_dump=False):
    
    print('-'*80)
    print('Test')
    
    step_observations = test_env.reset()
    step_terminal = numpy.ones(test_env.num_envs, dtype=numpy.bool)
    graph_states = [None] * test_env.num_envs
    for step in tqdm.tqdm(range(steps)):
        with torch.no_grad():
            # gym -> torch
            step_tensors = gym_space_to_tensors(
                    step_observations,
                    test_env.single_observation_space,
                    torch.cuda.current_device())
            
            # step model forward pass
            step_brick_lists, _, dense_scores, head_features = step_model(
                    step_tensors['color_render'],
                    step_tensors['segmentation_render'])
            
            # build new graph state for all terminal sequences
            # (do this here so we can use the brick_feature_spec from the model)
            for i, terminal in enumerate(step_terminal):
                if terminal:
                    graph_states[i] = BrickGraph(
                            BrickList(step_brick_lists[i].brick_feature_spec()),
                            edge_attr_channels=1).cuda()
            
            # upate the graph state using the edge model
            graph_states, _, _ = edge_model(step_brick_lists, graph_states)
            
            print([brick_list.num_nodes for brick_list in step_brick_lists])
            print([graph_state.num_nodes for graph_state in graph_states])
            edge_dicts = [graph.edge_dict() for graph in graph_states]
            
            print(edge_dicts)
            lkjlkjljklkj
            
            
            # sample an action
            hide_logits = [sbl.hide_action.view(-1) for sbl in step_brick_lists]
            hide_distributions = []
            bad_distributions = []
            for i, logits in enumerate(hide_logits):
                try:
                    distribution = Categorical(
                            logits=logits, validate_args=True)
                except ValueError:
                    bad_distrbutions.append(i)
                    distribution = Categorical(probs=torch.ones(1).cuda())
                hide_distributions.append(distribution)
            
            if len(bad_distributions):
                print('BAD DISTRIBUTIONS, REVERTING TO 0')
                print('STEP: %i, GLOBAL_STEP: %i'%(step, step_clock[0]))
                print('BAD INDICES:', bad_distributions)
                space = train_env.single_observation_space
                log_train_rollout_step(
                        step_clock, log, step_observations, space, [])
            
            segment_samples = [dist.sample() for dist in hide_distributions]
            #instance_samples = graph.remap_node_indices(
            #        batch_graphs, segment_samples, 'segment_id')
            instance_samples = [brick_list.segment_id[sample]
                    for brick_list, sample
                    in zip(step_brick_lists, segment_samples)]
            actions = [{'visibility':int(i.cpu())} for i in instance_samples]
            
            if step < log_debug:
                space = train_env.single_observation_space
                log_train_rollout_step(
                        step_clock,
                        log,
                        step_observations,
                        space,
                        actions)
            
            (step_observations,
             step_rewards,
             step_terminal,
             _) = train_env.step(actions)
            
            seq_rewards.append(step_rewards)
            
            step_clock[0] += 1










            
            
            instance_label_logits = batch_graphs.instance_label
            segment_index = batch_graphs.segment_index
            ptr = batch_graphs.ptr
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
            '''
            # sample an action
            # this is what we should be doing, but we're not training
            # visibility separately yet
            #hide_probs = torch.exp(batch_graph.hide_action.view(-1))
            #==========
            # this is what we are doing, reusing score (for now)
            hide_probs = batch_graph.score
            #hide_logits = torch.log(hide_probs / (1. - hide_probs))
            #==========
            #hide_distributions = graph.batch_graph_categoricals(
            #        batch_graph, logits=hide_logits)
            #segment_samples = [dist.sample() for dist in hide_distributions]
            split_probs = graph.split_node_value(batch_graph, hide_probs)
            segment_samples = [torch.argmax(p) for p in split_probs]
            instance_samples = graph.remap_node_indices(
                    batch_graph, segment_samples, 'segment_index')
            actions = [{'visibility':int(i.cpu())} for i in instance_samples]
            '''
            GET_THE_ABOVE_FROM_TRAIN
            
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
