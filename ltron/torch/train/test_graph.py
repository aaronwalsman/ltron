import os

import PIL.Image as Image

import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy

import renderpy.masks as masks

import brick_gym.evaluation as evaluation
import brick_gym.utils as utils
from brick_gym.dataset.paths import get_dataset_info
from brick_gym.gym.brick_env import async_brick_env
from brick_gym.gym.standard_envs import graph_supervision_env
from brick_gym.torch.brick_geometric import (
        BrickList, BrickGraph, BrickGraphBatch)
from brick_gym.torch.gym_tensor import (
        gym_space_to_tensors, gym_space_list_to_tensors, graph_to_gym_space)
import brick_gym.torch.models.named_models as named_models
from brick_gym.visualization.gym_dump import gym_dump
from brick_gym.torch.visualization.image_strip import make_image_strips

def test_checkpoint(
        # load checkpoints
        step_checkpoint,
        edge_checkpoint,
        
        # dataset settings
        dataset='random_stack',
        num_processes=8,
        test_split='test',
        test_subset=None,
        image_resolution=(256,256),
        
        # model settings
        step_model_name = 'nth_try',
        step_model_backbone = 'smp_fpn_r18',
        decoder_channels = 512,
        edge_model_name = 'squared_difference',
        segment_id_matching = False,
        
        # output settings
        dump_debug=False):
    
    device = torch.device('cuda:0')
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['class_ids'].values())+1
    max_instances_per_scene = dataset_info['max_instances_per_scene']
    
    if dump_debug:
        run = os.path.split(os.path.dirname(step_checkpoint))[-1]
        debug_directory = './debug/%s'%run
        if not os.path.exists(debug_directory):
            os.makedirs(debug_directory)
    else:
        debug_directory = None
    
    print('='*80)
    print('Building the step model')
    step_model = named_models.named_graph_step_model(
            step_model_name,
            backbone_name = step_model_backbone,
            decoder_channels = decoder_channels,
            num_classes = num_classes,
            input_resolution = image_resolution).cuda()
    step_model.load_state_dict(torch.load(step_checkpoint))
    
    print('-'*80)
    print('Building the edge model')
    edge_model = named_models.named_edge_model(
            edge_model_name,
            input_dim=decoder_channels).cuda()
    edge_model.load_state_dict(torch.load(edge_checkpoint))
    
    print('='*80)
    print('Building the test environment')
    if step_model_backbone == 'simple':
        segmentation_width, segmentation_height = (
                image_resolution[0] // 4, image_resolution[0] // 4)
    else:
        segmentation_width, segmentation_height = image_resolution
    test_env = async_brick_env(
            num_processes,
            graph_supervision_env,
            dataset = dataset,
            split=test_split,
            subset=test_subset,
            width = image_resolution[0],
            height = image_resolution[1],
            segmentation_width = segmentation_width,
            segmentation_height = segmentation_height,
            dataset_reset_mode = 'single_pass',
            multi_hide=True,
            randomize_viewpoint = False,
            randomize_viewpoint_frequency = 'reset',
            randomize_colors = False,
            random_floating_bricks=False)
            #random_floating_bricks=random_floating_bricks,
            #random_bricks_per_scene=random_bricks_per_scene,
            #random_bricks_subset=random_bricks_subset,
            #random_bricks_rotation_mode=random_bricks_rotation_mode)
    
    # test
    test_graph(
            step_model,
            edge_model,
            segment_id_matching,
            test_env,
            max_instances_per_scene,
            dump_debug,
            debug_directory)

def test_graph(
        # model
        step_model,
        edge_model,
        segment_id_matching,
        
        # environment
        test_env,
        
        max_instances_per_scene,
        
        # output
        dump_debug,
        debug_directory):
    
    device = torch.device('cuda:0')
    step_model.eval()
    edge_model.eval()
    #step_model.train()
    #edge_model.train()
    
    # get initial observations
    step_observations = test_env.reset()
    
    max_edges = test_env.single_action_space['graph_task'].max_edges
    
    # initialize progress variables
    step_terminal = numpy.ones(test_env.num_envs, dtype=numpy.bool)
    graph_states = [None] * test_env.num_envs
    all_finished = False
    scene_index = [ 0 for _ in range(test_env.num_envs)]
    
    # initialize statistic variables
    step_pred_edge_dicts = []
    step_pred_bland_edge_dicts = []
    step_target_edge_dicts = []
    step_target_bland_edge_dicts = []
    seq_target_edge_dicts = [{} for _ in range(test_env.num_envs)]
    seq_target_bland_edge_dicts = [{} for _ in range(test_env.num_envs)]
    
    #=========================== this is better =======================
    terminal_pred_edge_dict = {}
    terminal_pred_bland_edge_dict = {}
    terminal_pred_instance_dict = {}
    terminal_target_edge_dict = {}
    terminal_target_bland_edge_dict = {}
    terminal_target_instance_dict = {}
    prev_pred_edges = [{} for _ in range(test_env.num_envs)]
    prev_pred_bland_edges = [{} for _ in range(test_env.num_envs)]
    prev_pred_instances = [{} for _ in range(test_env.num_envs)]
    #==================================================================
    
    step_pred_instance_scores = []
    step_target_instance_labels = []
    seq_target_instance_labels = [{} for _ in range(test_env.num_envs)]
    
    # this is per-scene so it's wrong (a little bit)
    sum_terminal_instance_ap = 0.
    sum_terminal_edge_ap = 0.
    sum_terminal_bland_edge_ap = 0.
    total_terminal_ap = 0
    
    sum_all_instance_ap = 0.
    sum_all_edge_ap = 0.
    total_all_ap = 0
    
    max_instances_per_step = 1
    
    while not all_finished:
        with torch.no_grad():
            
            #-------------------------------------------------------------------
            # data storage and conversion
            # gym -> torch
            step_tensors = gym_space_to_tensors(
                    step_observations,
                    test_env.single_observation_space,
                    device)
            
            #-------------------------------------------------------------------
            # step model forward pass
            step_brick_lists, _, dense_scores, head_features = step_model(
                    step_tensors['color_render'],
                    step_tensors['segmentation_render'],
                    max_instances = max_instances_per_step)
            
            #-------------------------------------------------------------------
            # build new graph state for all terminal sequences
            # (do this here so we can use the brick_feature_spec from the model)
            # also update the target instance labels and graphs
            for i, terminal in enumerate(step_terminal):
                if terminal:
                    
                    #-----------------------------------------------------------
                    # convince the user that something is happening
                    if step_tensors['scene']['valid_scene_loaded'][i]:
                        print('Loading scene %i for process %i'%(
                                scene_index[i], i))
                    
                    #-----------------------------------------------------------
                    # make a new empty graph to represent the progress state
                    empty_brick_list = BrickList(
                            step_brick_lists[i].brick_feature_spec())
                    graph_states[i] = BrickGraph(
                            empty_brick_list, edge_attr_channels=1).cuda()
                    scene_index[i] += 1
                    
                    #-----------------------------------------------------------
                    # update the target instance labels
                    target_instance_labels = (
                            step_tensors['graph_label'][i].instance_label).cpu()
                    seq_target_instance_labels[i].clear()
                    for j in range(target_instance_labels.shape[0]):
                        label = int(target_instance_labels[j])
                        if label == 0:
                            continue
                        key = ((i, scene_index[i]), j, label)
                        seq_target_instance_labels[i][key] = 1.
                    terminal_target_instance_dict.update(
                        seq_target_instance_labels[i])
                    
                    #-----------------------------------------------------------
                    # update the target graph edges
                    target_edges = step_tensors['graph_label'][i].edge_index
                    unidirectional = target_edges[0] < target_edges[1]
                    target_edges = target_edges[:,unidirectional]
                    seq_target_edge_dicts[i].clear()
                    seq_target_bland_edge_dicts[i].clear()
                    for j in range(target_edges.shape[1]):
                        a = int(target_edges[0,j])
                        b = int(target_edges[1,j])
                        la = int(target_instance_labels[a])
                        lb = int(target_instance_labels[b])
                        key = ((i, scene_index[i]), a, b, la, lb)
                        seq_target_edge_dicts[i][key] = 1.
                        bland_key = ((i, scene_index[i]), a, b)
                        seq_target_bland_edge_dicts[i][bland_key] = 1.
                    terminal_target_edge_dict.update(
                            seq_target_edge_dicts[i])
                    terminal_target_bland_edge_dict.update(
                            seq_target_bland_edge_dicts[i])
            
            #-------------------------------------------------------------------
            # edge_model_forward_pass
            input_graph_states = graph_states
            graph_states, step_step_logits, step_state_logits = edge_model(
                    step_brick_lists,
                    graph_states,
                    max_edges=max_edges,
                    segment_id_matching=segment_id_matching)
            
            #-------------------------------------------------------------------
            # act
            hide_logits = [sbl.hide_action.view(-1) for sbl in step_brick_lists]
            '''
            #hide_logits = [sbl.score.view(-1) for sbl in step_brick_lists]
            hide_indices = []
            for i, logits in enumerate(hide_logits):
                if logits.shape[0]:
                    hide_value, segment = torch.max(logits, dim=0)
                    segment = int(segment.cpu())
                    hide_indices.append(
                            int(step_brick_lists[i].segment_id[segment].cpu()))
                else:
                    hide_indices.append(0)
            '''
            
            actions = []
            #for hide_index, graph_state in zip(hide_indices, graph_states):
            for i, logits in enumerate(hide_logits):
                if True: #multi_hide:
                    visibility_sample = numpy.zeros(
                            max_instances_per_scene+1, dtype=numpy.bool)
                    selected_instances = (
                            step_brick_lists[i].segment_id[:,0].cpu().numpy())
                    visibility_sample[selected_instances] = True
                else:
                    if logits.shape[0]:
                        hide_value, segment = torch.max(logits, dim=0)
                        segment = int(segment.cpu())
                        visibility_sample = int(
                                step_brick_lists[i].segment_id[segment_sample])
                
                graph_action = graph_to_gym_space(
                        graph_states[i].cpu(),
                        test_env.single_action_space['graph_task'],
                        process_instance_logits=True,
                        segment_id_remap=True)
                actions.append({
                        'visibility':visibility_sample,
                        'graph_task':graph_action})
            
            for i in range(test_env.num_envs):
                # report which edges were predicted this frame
                step = int(step_tensors['episode_length'][i].cpu())
                while len(step_pred_edge_dicts) <= step:
                    step_pred_edge_dicts.append({})
                    step_pred_bland_edge_dicts.append({})
                    step_target_edge_dicts.append({})
                    step_target_bland_edge_dicts.append({})
                    step_pred_instance_scores.append({})
                    step_target_instance_labels.append({})
                
                action = actions[i]['graph_task']
                edges = action['edges']['edge_index']
                scores = action['edges']['score']
                
                #unidirectional_edges = edges[0] < edges[1]
                #edges = edges[:,unidirectional_edges]
                #scores = action['edges']['score'][unidirectional_edges]
                
                pred_edges = utils.sparse_graph_to_edge_scores(
                        image_index = (i, scene_index[i]),
                        node_label = action['instances']['label'],
                        edges = edges.T,
                        scores = scores,
                        unidirectional = True)
                step_pred_edge_dicts[step].update(pred_edges)
                pred_bland_edges = utils.sparse_graph_to_edge_scores(
                        image_index = (i, scene_index[i]),
                        node_label = action['instances']['label'],
                        edges = edges.T,
                        scores = scores,
                        unidirectional = True,
                        include_node_labels = False)
                step_pred_bland_edge_dicts[step].update(pred_bland_edges)
                
                step_target_edge_dicts[step].update(seq_target_edge_dicts[i])
                step_target_bland_edge_dicts[step].update(
                        seq_target_bland_edge_dicts[i])
                
                if step_terminal[i]:
                    terminal_pred_edge_dict.update(prev_pred_edges[i])
                    terminal_pred_bland_edge_dict.update(
                            prev_pred_bland_edges[i])
                
                # ugh, I dislike this so much.
                prev_pred_edges[i] = pred_edges
                prev_pred_bland_edges[i] = pred_bland_edges
                
                #_, _, tmp_ap = evaluation.edge_ap(
                #        pred_edges, seq_target_edge_dicts[i])
                
                
                # report which instances were predicted this frame
                if graph_states[i].num_nodes:
                    indices = graph_states[i]['segment_id'].cpu().view(-1)
                    instance_label = torch.argmax(
                            graph_states[i].instance_label, dim=-1).cpu()
                    #scores = graph_states[i]['score'].cpu().view(-1)
                    
                    pred_instance_scores = (
                            utils.sparse_graph_to_instance_scores(
                            image_index = (i, scene_index[i]),
                            indices = indices.tolist(),
                            instance_labels = instance_label,
                            scores = scores.tolist()))
                    
                    step_pred_instance_scores[step].update(pred_instance_scores)
                    step_target_instance_labels[step].update(
                            seq_target_instance_labels[i])  
                    
                    if step_terminal[i]:
                        terminal_pred_instance_dict.update(
                                prev_pred_instances[i])
                    
                    # ugh, I dislike this too... so much
                    prev_pred_instances[i] = pred_instance_scores
            
            if dump_debug:
                '''
                gym_dump(
                        step_observations,
                        test_env.single_observation_space,
                        os.path.join(debug_directory, 'observation_'))
                '''
                instance_id = step_tensors['segmentation_render'].cpu().numpy()
                
                pred_class_labels = torch.argmax(
                        head_features['instance_label'], dim=1).cpu().numpy()
                target_class_labels = []
                for graph, seg in zip(step_tensors['graph_label'], instance_id):
                    instance_label = graph.instance_label[:,0]
                    target_class_labels.append(
                            instance_label[seg].cpu().numpy())
                
                step_segment_id = [brick_list.segment_id.cpu().numpy()
                        for brick_list in step_brick_lists]
                step_step_probs = [
                        torch.sigmoid(logits).cpu().numpy()
                        for logits in step_step_logits]
                step_step_match = [probs[...,0] for probs in step_step_probs]
                step_step_edge = [probs[...,1] for probs in step_step_probs]
                state_segment_id = [graph_state.segment_id.cpu().numpy()
                        for graph_state in input_graph_states]
                step_state_probs = [
                        torch.sigmoid(logits).cpu().numpy()
                        for logits in step_state_logits]
                step_state_match = [probs[...,0] for probs in step_state_probs]
                step_state_edge = [probs[...,1] for probs in step_state_probs]
                pred_image_strips = make_image_strips(
                        test_env.num_envs,
                        concatenate=False,
                        color_image = step_observations['color_render'],
                        dense_score = torch.sigmoid(dense_scores).cpu().numpy(),
                        dense_class_labels = pred_class_labels,
                        instance_id = instance_id,
                        step_size = [max_instances_per_step]*test_env.num_envs,
                        step_segment_ids = step_segment_id,
                        step_step_match = step_step_match,
                        step_step_edge = step_step_edge,
                        state_size = [120]*test_env.num_envs,
                        state_segment_ids = state_segment_id,
                        step_state_match = step_state_match,
                        step_state_edge = step_state_edge)
                
                correct = []
                for target, predicted in zip(
                        target_class_labels, pred_class_labels):
                    correct.append(target == predicted)
                
                step_step_match_target = [numpy.eye(ss.shape[0])
                        for ss in step_step_match]
                
                step_step_edge_target = []
                step_state_edge_target = []
                step_state_match_target = []
                for brick_list, pred_graph, target_graph in zip(
                        step_brick_lists,
                        input_graph_states,
                        step_tensors['graph_label']):
                    full_edge_target = target_graph.edge_matrix(
                            bidirectionalize=True).to(torch.bool).cpu().numpy()
                    step_lookup = brick_list.segment_id[:,0].cpu().numpy()
                    edge_target = full_edge_target[
                            step_lookup][:,step_lookup]
                    step_step_edge_target.append(edge_target)
                    
                    #state_lookup = graph.segment_id[:,0].cpu().numpy()
                    #state_lookup = numpy.array(range(target_graph.num_nodes))
                    #state_lookup = state_lookup[
                    #        pred_graph.segment_id[:,0].cpu().numpy()]
                    state_lookup = pred_graph.segment_id[:,0].cpu().numpy()
                    edge_target = full_edge_target[
                            step_lookup][:,state_lookup]
                    step_state_edge_target.append(edge_target)
                    
                    step_lookup = step_lookup.reshape(step_lookup.shape[0], 1)
                    state_lookup = state_lookup.reshape(
                            1, state_lookup.shape[0])
                    match_target = step_lookup == state_lookup
                    step_state_match_target.append(match_target)
                
                target_image_strips = make_image_strips(
                        test_env.num_envs,
                        concatenate = False,
                        color_image = step_observations['color_render'],
                        dense_score = correct,
                        dense_class_labels = target_class_labels,
                        instance_id = instance_id,
                        step_size = [max_instances_per_step]*test_env.num_envs,
                        step_segment_ids = step_segment_id,
                        step_step_match = step_step_match_target,
                        step_step_edge = step_step_edge_target,
                        state_size = [120]*test_env.num_envs,
                        state_segment_ids = state_segment_id,
                        step_state_match = step_state_match_target,
                        step_state_edge = step_state_edge_target)
                
                for i, (pred_strip, target_strip) in enumerate(
                        zip(pred_image_strips, target_image_strips)):
                    if step_tensors['scene']['valid_scene_loaded'][i]:
                        scene_id = step_tensors['dataset'][i]
                        step_id = step_tensors['episode_length'][i]
                        pred_strip_path = os.path.join(
                                debug_directory,
                                'pred_strip_%i_%i.png'%(scene_id, step_id))
                        Image.fromarray(pred_strip).save(pred_strip_path)
                        
                        target_strip_path = os.path.join(
                                debug_directory,
                                'target_strip_%i_%i.png'%(scene_id, step_id))
                        Image.fromarray(target_strip).save(target_strip_path)
            
            # take actions
            (step_observations,
             step_rewards,
             step_terminal,
             info) = test_env.step(actions)
            
            for i,t in enumerate(step_terminal):
                # use step-tensors because it's one step out of date, and so
                # we won't end up clipping the last one.
                if step_tensors['scene']['valid_scene_loaded'][i]:
                    if t:
                        #print('edge/instance ap:')
                        #print(info[i]['graph_task']['edge_ap'])
                        #print(info[i]['graph_task']['instance_ap'])
                        sum_terminal_instance_ap += (
                                info[i]['graph_task']['instance_ap'])
                        sum_terminal_edge_ap += (
                                info[i]['graph_task']['edge_ap'])
                        total_terminal_ap += 1
                    
                    sum_all_instance_ap += (
                            info[i]['graph_task']['instance_ap'])
                    sum_all_edge_ap += (
                            info[i]['graph_task']['edge_ap'])
                    total_all_ap += 1
            
            # done yet?
            all_finished = numpy.all(
                    step_tensors['scene']['valid_scene_loaded'] == 0)
            
    print('-'*80)
    print('Edge AP: regular (bland)')
    edge_ap_values = []
    for step, (
            pred_edge_dict,
            pred_bland,
            target_edge_dict,
            target_bland) in enumerate(zip(
            step_pred_edge_dicts,
            step_pred_bland_edge_dicts,
            step_target_edge_dicts,
            step_target_bland_edge_dicts)):
        _, _, edge_ap = evaluation.edge_ap(pred_edge_dict, target_edge_dict)
        _, _, bland_edge_ap = evaluation.edge_ap(pred_bland, target_bland)
        edge_ap_values.append(edge_ap)
        print('  Step %i: %f (%f)'%(step, edge_ap, bland_edge_ap))
    
    _, _, edge_ap = evaluation.edge_ap(
            terminal_pred_edge_dict, terminal_target_edge_dict)
    _, _, bland_edge_ap = evaluation.edge_ap(
            terminal_pred_bland_edge_dict, terminal_target_bland_edge_dict)
    print('Terminal: %f (%f)'%(edge_ap, bland_edge_ap))
    
    
    '''
    import matplotlib.pyplot as pyplot
    pyplot.plot(edge_ap_values)
    pyplot.ylim(0, 1)
    pyplot.show()
    '''
    print('-'*80)
    print('Instance AP:')
    instance_ap_values = []
    for step, (pred_instance_dict, target_instance_dict) in enumerate(zip(
                step_pred_instance_scores, step_target_instance_labels)):
        _, _, instance_ap = evaluation.edge_ap(
                pred_instance_dict, target_instance_dict)
        instance_ap_values.append(instance_ap)
        print('  Step %i: %f'%(step, instance_ap))
    _, _, instance_ap = evaluation.edge_ap(
            terminal_pred_instance_dict, terminal_target_instance_dict)
    print('Terminal: %f'%instance_ap)
    
    '''
    import matplotlib.pyplot as pyplot
    pyplot.plot(instance_ap_values)
    pyplot.ylim(0, 1)
    pyplot.show()
    '''
    
    print('='*80)
    print('Single Step Stats:')
    
    print('Average instance ap: %f'%(
            sum_all_instance_ap/total_all_ap))
    print('Average edge ap: %f'%(
            sum_all_edge_ap/total_all_ap))
    print('Average terminal instance ap: %f'%(
            sum_terminal_instance_ap/total_terminal_ap))
    print('Average terminal edge ap: %f'%(
            sum_terminal_edge_ap/total_terminal_ap))
    
    # statistics I want:
    # confident node-labelling accuracy (PR-curve)
    #   record each classification, corresponding confidence and ground-truth
    # edge AP over entire dataset for each step using predicted matching
    #   record predicted and ground truth edge dictionaries
    # edge AP over entire dataset using ground-truth matching
    #   "
    # edge AP over entire dataset using ground-truth matching and edges
    #   "
    # for both train and test set
    #
    # note that per-step rewards are not a good enough signal for any of this
