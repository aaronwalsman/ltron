import math
import os

import torch
from torchvision.models import resnet18
import torch.distributions as distributions

import numpy

import PIL.Image as Image

import tqdm

from brick_gym.torch.utils import segments_to_tensor, image_segments_to_tensor
from brick_gym.multiclass import MultiClass
from brick_gym.envs.graph_env import GraphEnv
from brick_gym.viewpoint.azimuth_elevation import FixedAzimuthalViewpoint
from brick_gym.torch.models.action import ActionModel, FeatureActionModel
from brick_gym.torch.models.graph_accumulator import GraphAccumulator
import brick_gym.evaluation as evaluation
import brick_gym.torch.models.resnet as bg_resnet
import brick_gym.torch.models.standard_models as standard_models
import brick_gym.utils as utils
import brick_gym.torch.utils as bgt_utils

EPS = 1e-5

tmp_root_dir = ('/media/awalsman/data_drive/brick-gym/brick_gym/'
        'random_stack/online_edges_ground_truth5')
confidence_path = os.path.join(
        tmp_root_dir, 'confidence_checkpoint_0300.pt')
edge_path = os.path.join(
        tmp_root_dir, 'edge_checkpoint_0300.pt')
model_path = os.path.join(
        tmp_root_dir, 'model_checkpoint_0300.pt')
segmentation_path = os.path.join(
        tmp_root_dir, 'segmentation_checkpoint_0300.pt')

def train_label_confidence(
        num_epochs,
        dataset,
        train_split = 'train_mpd',
        test_split = 'test_mpd',
        num_processes = 8,
        learning_rate = 3e-4,
        train_episodes_per_epoch = 256,
        checkpoint_frequency = 1)

def train_hide_reinforce(
        num_epochs,
        dataset,
        train_split = 'train_mpd',
        test_split = 'test_mpd',
        num_processes = 16,
        learning_rate = 3e-2,
        discount = 0.9,
        train_episodes_per_epoch = 256,
        input_mode = 'features',
        checkpoint_frequency = 1):
    
    print('Loading labelling models')
    feature_model = resnet18().cuda()
    bg_resnet.make_spatial_attention_resnet(
            feature_model, shape=(256,256), do_spatial_embedding=True)
    bg_resnet.replace_fc(feature_model, 512)
    feature_model.load_state_dict(torch.load(model_path))
    
    brick_classifier = torch.nn.Linear(512, 7).cuda()
    brick_classifier.load_state_dict(torch.load(segmentation_path))
    
    confidence_classifier = torch.nn.Linear(512, 2).cuda()
    confidence_classifier.load_state_dict(torch.load(confidence_path))
    
    edge_classifier = standard_models.get_edge_model('simple_edge_512').cuda()
    edge_classifier.load_state_dict(torch.load(edge_path))
    
    graph_model = GraphAccumulator(
            feature_model,
            brick_classifier,
            confidence_classifier,
            edge_classifier)
    
    print('Building action model')
    if input_mode == 'features':
        action_model = FeatureActionModel().cuda()
    elif input_mode == 'images':
        action_model = ActionModel().cuda()
    elif input_mode == 'confidence':
        action_model = FeatureActionModel(2, bias=False).cuda()
        #action_model.model.weight = torch.nn.Parameter(torch.FloatTensor(
        #        [[-1,1]]).cuda())
    
    print('Building environments')
    viewpoint_control = FixedAzimuthalViewpoint(
            azimuth = math.radians(30.), elevation = -math.radians(45))
    train_multi_env = MultiClass(
            num_processes,
            GraphEnv,
            [{'dataset' : dataset,
              'split' : train_split,
              'viewpoint_control' : viewpoint_control,
              'rank' : i,
              'size' : num_processes,
              'reward_mode' : 'edge_ap',
              'reset_mode' : 'random'} for i in range(num_processes)])
    
    test_multi_env = MultiClass(
            num_processes,
            GraphEnv,
            [{'dataset' : dataset,
              'split' : test_split,
              'viewpoint_control' : viewpoint_control,
              'rank' : i,
              'size' : num_processes,
              'reward_mode' : 'edge_ap'} for i in range(num_processes)])
    
    print('Building optimizer')
    optimizer = torch.optim.Adam(action_model.parameters(), lr=learning_rate)
    
    print('Starting processes')
    with train_multi_env, test_multi_env:
        for epoch in range(1, num_epochs+1):
            print('Epoch: %i/%i'%(epoch, num_epochs))
            
            step_rewards = None
            
            iterate = tqdm.tqdm(
                    range(0, train_episodes_per_epoch, num_processes))
            for i in iterate:
                
                # initialize data storage
                batch_rewards = []
                batch_action_logps = []
                terminal = [False] * num_processes
                
                # reset environment
                observations = train_multi_env.call_method('reset')
                graph_model.reset()
                batch_images, batch_segments = zip(*observations)
                node_edge_labels = train_multi_env.call_method(
                        'get_node_and_edge_labels')
                node_labels, edge_labels = zip(*node_edge_labels)
                batch_edge_labels = []
                for edges in edge_labels:
                    batch_edge_labels.append(
                            {(a-1, b-1, c, d):1.0 for a,b,c,d in edges})
                
                with torch.no_grad():
                    x_s = bgt_utils.segments_to_tensor(
                            batch_segments).cuda()
                    (step_features,
                     step_confidence_logits,
                     step_node_logits,
                     progress_node_logits,
                     progress_edge_logits) = graph_model(x_s)
                
                # rollout
                step = 0
                while not all(terminal):
                    #print('------')
                    '''
                    for j, process_obs in enumerate(observations):
                        Image.fromarray(process_obs[0]).save(
                                './image_%i_%i_%i.png'%(i,j,step))
                        for k, step_obs in enumerate(process_obs[1]):
                            Image.fromarray(step_obs).save(
                                    './masked_image_%i_%i_%i_%i.png'%(
                                        i,j,step,k))
                    '''
                    if input_mode == 'features':
                        action_logits = action_model(step_features)
                    elif input_mode == 'confidence':
                        #r = torch.rand((16, 8, 1)).cuda() * 2 - 1.0
                        #x = torch.cat((r, step_confidence_logits), dim=-1)
                        action_logits = action_model(step_confidence_logits)
                        #print(action_model.model.weight)
                    else:
                        # process data
                        x_is = bgt_utils.image_segments_to_tensor(
                                batch_images, batch_segments).cuda()
                        
                        # compute action
                        action_logits = action_model(x_is)
                    
                    # sample actions
                    predicted_zero = torch.argmax(step_node_logits, dim=-1) == 0
                    #print('predicted zero:')
                    #print(predicted_zero)
                    action_logits = (action_logits * ~predicted_zero + 
                            -100000. * predicted_zero)
                    
                    #======
                    action_distribution = distributions.Categorical(
                            logits = action_logits)
                    actions = action_distribution.sample()
                    #actions = torch.argmax(action_logits, dim=-1)
                    #======
                    
                    #TEMP
                    '''
                    confidence = torch.softmax(step_confidence_logits, dim=-1)
                    confidence = confidence[:,:,-1]
                    confidence = (confidence * ~predicted_zero +
                            0. * predicted_zero)
                    actions = torch.argmax(confidence, dim=-1)
                    '''
                    #print(actions)
                    #print(node_labels)
                    #print(action_logits)
                    #print(step_confidence_logits)
                    #print(actions)
                    batch_action_logps.append(
                            action_distribution.log_prob(actions))
                    
                    # step
                    batch_actions = []
                    for j in range(num_processes):
                        batch_actions.append({'action':{
                                'hide' : int(actions[j].cpu())}})
                    # do not use the "official" official reward
                    observations, _, terminal, _ = zip(
                            *train_multi_env.call_method('step', batch_actions))
                    batch_images, batch_segments = zip(*observations)
                    
                    # compute new reward based on labelling networks
                    with torch.no_grad():
                        x_s = bgt_utils.segments_to_tensor(
                                batch_segments).cuda()
                        (step_features,
                         step_confidence_logits,
                         step_node_logits,
                         progress_node_logits,
                         progress_edge_logits) = graph_model(x_s)
                        progress_nodes = torch.argmax(
                                progress_node_logits, dim=-1)
                        progress_edges = torch.sigmoid(progress_edge_logits)
                        
                        rewards = []
                        for j, batch_edges in enumerate(batch_edge_labels):
                            predicted_edges = utils.matrix_to_edge_scores(
                                    None, progress_nodes[j], progress_edges[j])
                            _, _, reward = evaluation.edge_ap(
                                    predicted_edges, batch_edges)
                            rewards.append(reward)
                        
                        batch_rewards.append(rewards)
                        #print(rewards)
                        #print(sum(rewards)/len(rewards))
                    
                    step += 1
                
                # compute returns
                returns = []
                prev_rewards = [0] * num_processes
                for rewards in batch_rewards[::-1]:
                    returns.append([
                            r+discount*p
                            for r,p in zip(rewards, prev_rewards)])
                    prev_rewards = rewards
                returns = torch.FloatTensor(list(reversed(returns)))
                
                # normalize using mean/variance
                s, b = returns.shape
                mean = torch.mean(returns, dim=-1)
                std = torch.std(returns, dim=-1)
                norm_returns = (
                        (returns - mean.unsqueeze(-1)) /
                        (std.unsqueeze(-1) + EPS)).cuda()
                #norm_returns = (returns - returns.mean()) / returns.std() + EPS
                #norm_returns = norm_returns.cuda()
                
                # compute loss
                batch_action_logps = torch.stack(batch_action_logps)
                loss = torch.mean(-batch_action_logps * norm_returns)
                
                # backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # description
                iterate.set_description('Loss: %.04f'%(float(loss)))
                
                reward_sum = numpy.sum(numpy.array(batch_rewards), axis=-1)
                if step_rewards is None:
                    step_rewards = reward_sum
                else:
                    step_rewards += reward_sum
            
            for step, reward in enumerate(step_rewards):
                print('Average reward at step %i: %f'%(
                        step, reward/train_episodes_per_epoch))
            
            if epoch % checkpoint_frequency == 0:
                checkpoint_path = './action_%04i.pt'%epoch
                print('Saving checkpoint to: %s'%checkpoint_path)
                torch.save(action_model.state_dict(), checkpoint_path)

def test_hide_reinforce(
        test_checkpoint,
        dataset,
        test_split = 'test_mpd',
        subset = None,
        input_mode = 'images',
        num_processes = 16):
    
    print('Loading labelling models')
    feature_model = resnet18().cuda()
    bg_resnet.make_spatial_attention_resnet(
            feature_model, shape=(256,256), do_spatial_embedding=True)
    bg_resnet.replace_fc(feature_model, 512)
    feature_model.load_state_dict(torch.load(model_path))
    feature_model.eval()
    
    brick_classifier = torch.nn.Linear(512, 7).cuda()
    brick_classifier.load_state_dict(torch.load(segmentation_path))
    brick_classifier.eval()
    
    confidence_classifier = torch.nn.Linear(512, 2).cuda()
    confidence_classifier.load_state_dict(torch.load(confidence_path))
    confidence_classifier.eval()
    
    edge_classifier = standard_models.get_edge_model('simple_edge_512').cuda()
    edge_classifier.load_state_dict(torch.load(edge_path))
    edge_classifier.eval()
    '''
    graph_model = GraphAccumulator(
            feature_model,
            brick_classifier,
            confidence_classifier,
            edge_classifier)
    '''
    print('Building action model')
    if input_mode == 'features':
        action_model = FeatureActionModel().cuda()
    elif input_mode == 'images':
        action_model = ActionModel().cuda()
    elif input_mode == 'confidence':
        action_model = FeatureActionModel(2, bias=False).cuda()
    action_model.load_state_dict(torch.load(test_checkpoint))
    action_model.eval()
    
    class ModelWrapper():
        def __call__(self, observations, hidden_state):
            #observations = [segs for im, segs in observations]
            batch_images, batch_segments = zip(*observations)
            x = bgt_utils.segments_to_tensor(batch_segments).cuda()
            ep, inst, ch, h, w = x.shape
            x = feature_model(x.view(ep*inst, ch, h, w))
            node_logits = brick_classifier(x)
            confidence_logits = confidence_classifier(x)
            
            step_node_predictions = torch.argmax(
                    node_logits.view(ep, inst, -1), dim=-1)
            
            x = x.view(ep, inst, -1)
            if hidden_state is None:
                hidden_state = (
                        torch.zeros_like(step_node_predictions).cpu(),
                        torch.zeros_like(x).cpu())
            
            label_accumulator, feature_accumulator = hidden_state
            
            if input_mode == 'features':
                action_logits = action_model(x)
            elif input_mode == 'confidence':
                #r = torch.rand((16, 8, 1)).cuda() * 2 - 1.0
                #x = torch.cat((r, step_confidence_logits), dim=-1)
                action_logits = action_model(confidence_logits)
                #print(action_model.model.weight)
            else:
                # process data
                x_is = bgt_utils.image_segments_to_tensor(
                        batch_images, batch_segments).cuda()
                
                # compute action
                action_logits = action_model(x_is)
            
            predicted_zero = step_node_predictions == 0
            #print('predicted zero:')
            #print(predicted_zero)
            action_logits = (torch.rand(action_logits.shape)*100. - 50.).cuda()
            action_logits = (action_logits * ~predicted_zero + 
                    -100000. * predicted_zero)
            action = torch.argmax(action_logits, dim=-1)
            #action = torch.randint(6, action.shape).cuda()
            '''
            confidence = torch.softmax(confidence_logits, dim=-1)[:,1]
            confidence = confidence.view(ep, inst)
            confidence = confidence * (step_node_predictions != 0)
            action = torch.argmax(confidence, dim=-1)
            '''
            
            #if args.accumulator_mode == 'argmax':
            if False:
                label_accumulator[range(ep),action] = (
                        step_node_predictions[range(ep),action].cpu())
                feature_accumulator[range(ep),action] = (
                        x[range(ep),action].cpu())
            #elif args.accumulator_mode == 'all':
            else:
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
    
    print('Building environments')
    viewpoint_control = FixedAzimuthalViewpoint(
            azimuth = math.radians(30.), elevation = -math.radians(45))
    
    test_multi_env = MultiClass(
            num_processes,
            GraphEnv,
            [{'dataset' : dataset,
              'split' : test_split,
              'subset' : subset,
              'viewpoint_control' : viewpoint_control,
              'rank' : i,
              'size' : num_processes,
              'reward_mode' : 'edge_ap',
              'reset_mode' : 'sequential'} for i in range(num_processes)])
    
    with test_multi_env, torch.no_grad():
        m = ModelWrapper()
        step_ap = evaluation.dataset_node_and_edge_ap(
                m, test_multi_env, dump_images=False)

def test_best(
        test_checkpoint,
        dataset,
        test_split = 'test_mpd',
        subset = None,
        input_mode = 'images',
        num_processes = 16):
    
    print('Loading labelling models')
    feature_model = resnet18().cuda()
    bg_resnet.make_spatial_attention_resnet(
            feature_model, shape=(256,256), do_spatial_embedding=True)
    bg_resnet.replace_fc(feature_model, 512)
    feature_model.load_state_dict(torch.load(model_path))
    
    brick_classifier = torch.nn.Linear(512, 7).cuda()
    brick_classifier.load_state_dict(torch.load(segmentation_path))
    
    confidence_classifier = torch.nn.Linear(512, 2).cuda()
    confidence_classifier.load_state_dict(torch.load(confidence_path))
    
    edge_classifier = standard_models.get_edge_model('simple_edge_512').cuda()
    edge_classifier.load_state_dict(torch.load(edge_path))
    
    class ModelWrapper():
        def __call__(self, observations, hidden_state):
            batch_images, batch_segments = zip(*observations)
            x = bgt_utils.segments_to_tensor(batch_segments).cuda()
            ep, inst, ch, h, w = x.shape
            x = feature_model(x.view(ep*inst, ch, h, w))
            node_logits = brick_classifier(x)
            confidence_logits = confidence_classifier(x)
            
            step_node_predictions = torch.argmax(
                    node_logits.view(ep, inst, -1), dim=-1)
            
            x = x.view(ep, inst, -1)
            if hidden_state is None:
                hidden_state = (
                        torch.zeros_like(step_node_predictions).cpu(),
                        torch.zeros_like(x).cpu())
            
            label_accumulator, feature_accumulator = hidden_state
            
            if input_mode == 'features':
                action_logits = action_model(x)
            elif input_mode == 'confidence':
                #r = torch.rand((16, 8, 1)).cuda() * 2 - 1.0
                #x = torch.cat((r, step_confidence_logits), dim=-1)
                action_logits = action_model(confidence_logits)
                #print(action_model.model.weight)
            else:
                # process data
                x_is = bgt_utils.image_segments_to_tensor(
                        batch_images, batch_segments).cuda()
                
                # compute action
                action_logits = action_model(x_is)
            
            predicted_zero = step_node_predictions == 0
            action_logits = (action_logits * ~predicted_zero + 
                    -100000. * predicted_zero)
            action = torch.argmax(action_logits, dim=-1)
            
            #if args.accumulator_mode == 'argmax':
            if False:
                label_accumulator[range(ep),action] = (
                        step_node_predictions[range(ep),action].cpu())
                feature_accumulator[range(ep),action] = (
                        x[range(ep),action].cpu())
            #elif args.accumulator_mode == 'all':
            else:
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
    
    print('Building environments')
    viewpoint_control = FixedAzimuthalViewpoint(
            azimuth = math.radians(30.), elevation = -math.radians(45))
    
    test_multi_env = MultiClass(
            num_processes,
            GraphEnv,
            [{'dataset' : dataset,
              'split' : test_split,
              'subset' : subset,
              'viewpoint_control' : viewpoint_control,
              'rank' : i,
              'size' : num_processes,
              'reward_mode' : 'edge_ap',
              'reset_mode' : 'sequential'} for i in range(num_processes)])
    
    with test_multi_env, torch.no_grad():
        m = ModelWrapper()
        step_ap = evaluation.dataset_node_and_edge_ap(
                m, test_multi_env, dump_images=False)
