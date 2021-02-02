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

def test_checkpoint(
        # load checkpoints
        step_checkpoint,
        edge_checkpoint,
        
        # dataset settings
        dataset='random_stack',
        num_processes=8,
        test_split='test',
        test_subset=None,
        
        # model settings
        step_model_name = 'nth_try',
        edge_model_name = 'subtract',
        segment_id_matching = False):
    
    device = torch.device('cuda:0')
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['class_ids'].values())+1
    
    print('='*80)
    print('Building the step model')
    step_model = named_models.named_graph_step_model(
            step_model_name,
            num_classes = num_classes).cuda()
    step_model.load_state_dict(torch.load(step_checkpoint))
    step_model.eval()
    
    print('-'*80)
    print('Building the edge model')
    edge_model = named_models.named_edge_model(
            edge_model_name,
            input_dim=256).cuda()
    edge_model.load_state_dict(torch.load(edge_checkpoint))
    edge_model.eval()
    
    print('='*80)
    print('Building the test environment')
    test_env = async_brick_env(
            num_processes,
            graph_supervision_env,
            dataset = dataset,
            split=test_split,
            subset=test_subset,
            dataset_reset_mode = 'single_pass',
            randomize_viewpoint = False,
            randomize_viewpoint_frequency = 'reset',
            randomize_colors = False)
    
    # test
    test_graph(
            step_model,
            edge_model,
            segment_id_matching,
            test_env)

def test_graph(
        # model
        step_model,
        edge_model,
        segment_id_matching,
        
        # environment
        test_env):
    
    device = torch.device('cuda:0')
    
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
    step_target_edge_dicts = []
    seq_target_edge_dicts = [{} for _ in range(test_env.num_envs)]
    
    step_pred_instance_scores = []
    step_target_instance_labels = []
    seq_target_instance_labels = [{} for _ in range(test_env.num_envs)]
    
    sum_terminal_instance_ap = 0.
    sum_terminal_edge_ap = 0.
    total_terminal_ap = 0
    
    sum_all_instance_ap = 0.
    sum_all_edge_ap = 0.
    total_all_ap = 0
    
    while not all_finished:
        with torch.no_grad():
            # gym -> torch
            step_tensors = gym_space_to_tensors(
                    step_observations,
                    test_env.single_observation_space,
                    device)
            
            # step model forward pass
            step_brick_lists, _, dense_scores, head_features = step_model(
                    step_tensors['color_render'],
                    step_tensors['segmentation_render'])
            
            # build new graph state for all terminal sequences
            # (do this here so we can use the brick_feature_spec from the model)
            # also update the target instance labels and graphs
            for i, terminal in enumerate(step_terminal):
                if terminal:
                    
                    # print something
                    if step_tensors['scene']['valid_scene_loaded'][i]:
                        print('Loading scene %i for process %i'%(
                                scene_index[i], i))
                    
                    # make a new empty graph to represent the progress state
                    empty_brick_list = BrickList(
                            step_brick_lists[i].brick_feature_spec())
                    graph_states[i] = BrickGraph(
                            empty_brick_list, edge_attr_channels=1).cuda()
                    scene_index[i] += 1
                    
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
                    
                    # update the target graph edges
                    target_edges = step_tensors['graph_label'][i].edge_index
                    unidirectional = target_edges[0] < target_edges[1]
                    target_edges = target_edges[:,unidirectional]
                    seq_target_edge_dicts[i].clear()
                    for j in range(target_edges.shape[1]):
                        a = int(target_edges[0,j])
                        b = int(target_edges[1,j])
                        la = int(target_instance_labels[a])
                        lb = int(target_instance_labels[b])
                        key = ((i, scene_index[i]), a, b, la, lb)
                        seq_target_edge_dicts[i][key] = 1.
            
            # update the graph state using the edge model
            graph_states, step_edge_logits, step_state_logits = edge_model(
                    step_brick_lists,
                    graph_states,
                    segment_id_matching = segment_id_matching,
                    max_edges=max_edges)
            
            # figure out which instances to hide
            hide_logits = [sbl.hide_action.view(-1) for sbl in step_brick_lists]
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
            
            # construct the action
            actions = []
            for hide_index, graph_state in zip(hide_indices, graph_states):
                graph_action = graph_to_gym_space(
                        graph_state.cpu(),
                        test_env.single_action_space['graph_task'],
                        process_instance_logits=True,
                        segment_id_remap=True)
                actions.append({
                        'visibility':hide_index,
                        'graph_task':graph_action})
            
            for i in range(test_env.num_envs):
                '''
                # spit out debug images
                s = step_tensors['episode_length'].cpu()[i]
                color_image = step_observations['color_render'][i]
                Image.fromarray(color_image).save(
                        './tmp_color_%i_%i_%i.png'%(i, scene_index[i], s))
                
                mask_image = step_observations['segmentation_render'][0]
                mask = masks.color_index_to_byte(mask_image)
                Image.fromarray(mask).save(
                        './tmp_mask_%i_%i_%i.png'%(i, scene_index[i], s))
                
                h, w = color_image.shape[:2]
                score_image = (dense_scores[i].view(h, w, 1).expand(
                        h, w, 3).cpu().numpy() * 255).astype(numpy.uint8)
                Image.fromarray(score_image).save(
                        './score_%i_%i_%i.png'%(i, scene_index[i], s))
                '''
                
                # report which edges were predicted this frame
                step = int(step_tensors['episode_length'][i].cpu())
                while len(step_pred_edge_dicts) <= step:
                    step_pred_edge_dicts.append({})
                    step_target_edge_dicts.append({})
                    step_pred_instance_scores.append({})
                    step_target_instance_labels.append({})
                
                action = actions[i]['graph_task']
                edges = action['edges']['edge_index']
                unidirectional_edges = edges[0] < edges[1]
                
                edges = edges[:,unidirectional_edges]
                scores = action['edges']['score'][unidirectional_edges]
                
                pred_edges = utils.sparse_graph_to_edge_scores(
                        image_index = (i, scene_index[i]),
                        node_label = action['instances']['label'],
                        edges = edges.T,
                        scores = scores)
                step_pred_edge_dicts[step].update(pred_edges)
                step_target_edge_dicts[step].update(seq_target_edge_dicts[i])
                
                # report which instances were predicted this frame
                if graph_states[i].num_nodes:
                    indices = graph_states[i]['segment_id'].cpu().view(-1)
                    instance_label = torch.argmax(
                            graph_states[i].instance_label, dim=-1).cpu()
                    scores = graph_states[i]['score'].cpu().view(-1)
                    pred_instance_scores = (
                            utils.sparse_graph_to_instance_scores(
                            image_index = (i, scene_index[i]),
                            indices = indices.tolist(),
                            instance_labels = instance_label,
                            scores = scores.tolist()))
                
                    step_pred_instance_scores[step].update(pred_instance_scores)
                    step_target_instance_labels[step].update(
                            seq_target_instance_labels[i])
            
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
    print('Edge AP:')
    edge_ap_values = []
    for step, (pred_edge_dict, target_edge_dict) in enumerate(zip(
                step_pred_edge_dicts, step_target_edge_dicts)):
        _, _, edge_ap = evaluation.edge_ap(pred_edge_dict, target_edge_dict)
        edge_ap_values.append(edge_ap)
        print('  Step %i: %f'%(step, edge_ap))
    
    import matplotlib.pyplot as pyplot
    pyplot.plot(edge_ap_values)
    pyplot.ylim(0, 1)
    pyplot.show()
    
    print('-'*80)
    print('Instance AP:')
    instance_ap_values = []
    for step, (pred_instance_dict, target_instance_dict) in enumerate(zip(
                step_pred_instance_scores, step_target_instance_labels)):
        _, _, instance_ap = evaluation.edge_ap(
                pred_instance_dict, target_instance_dict)
        instance_ap_values.append(instance_ap)
        print('  Step %i: %f'%(step, instance_ap))
    
    import matplotlib.pyplot as pyplot
    pyplot.plot(instance_ap_values)
    pyplot.ylim(0, 1)
    pyplot.show()
    
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
