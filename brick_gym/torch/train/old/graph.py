import math

import torch
from torchvision.transforms.functional import to_tensor

import PIL.Image as Image

import tqdm

import brick_gym.config as config
from brick_gym.multiclass import MultiClass
from brick_gym.envs.graph_env import GraphEnv
from brick_gym.viewpoint.azimuth_elevation import FixedAzimuthalViewpoint
import brick_gym.torch.models.standard_models as standard_models

EPS = 1e-5

def step_supervision_losses(
        label_logits,
        confidence_logits,
        edge_logits,
        label_targets,
        edges_targets):
    
    # label loss
    label_loss = torch.nn.functional.cross_entropy(label_logits, label_targets)
    
    # confidence_loss
    predicted_labels = torch.argmax(label_logits, dim=-1)
    
    # edge loss
    edge_loss = torch.nn.functional.cross_entropy(
            edge_logits.view(-1,2),
            edge_targets.view(-1),
            weight = edge_weights,
            reduction = 'none').view(bs, nodes, nodes)
    
    return label_loss, confidence_loss, edge_loss

def alternate_reinforce_supervise(
        num_epochs,
        num_processes,
        train_episodes_per_epoch,
        test_episodes_per_epoch,
        mini_epochs,
        dataset,
        node_model_name,
        edge_model_name,
        discount,
        learning_rate=3e-4,
        batch_size = 32,
        confidence_threshold=0.5,
        test_frequency=1,
        checkpoint_frequency=1):
    
    print('Building node model')
    node_model = standard_models.get_graphaction_model(
            node_model_name, classes=7).cuda()
    print('Building edge model')
    edge_model = standard_models.get_edge_model(edge_model_name).cuda()
    
    print('Building optimizer')
    optimizer = torch.optim.Adam(
            list(node_model.parameters()) + list(edge_model.parameters()),
            lr = learning_rate)
    
    print('Building environment')
    viewpoint_control = FixedAzimuthalViewpoint(
            azimuth = math.radians(30.), elevation = -math.radians(45))
    train_multi_env = MultiClass(
            num_processes,
            GraphEnv,
            [{'dataset':dataset,
              'split':'train_mpd',
              'viewpoint_control':viewpoint_control,
              'rank':i,
              'size':num_processes,
              'reward_mode':'node_accuracy'} for i in range(num_processes)])
    
    num_test_processes = 1
    test_multi_env = MultiClass(
            num_test_processes,
            GraphEnv,
            [{'dataset':dataset,
              'split':'test_mpd',
              'viewpoint_control':viewpoint_control,
              'rank':i,
              'size':num_test_processes,
              'reward_mode':'node_accuracy'} for i in range(num_processes)])
    
    with train_multi_env, test_multi_env:
        max_instances = train_multi_env.get_attr(
                'max_instances', processes=[0])[0]
        
        for epoch in range(1, num_epochs+1):
            print('RL Epoch: %i'%epoch)
            
            # initialize data storage
            epoch_observations = []
            epoch_hidden_nodes = []
            epoch_node_labels = []
            epoch_edge_labels = []
            epoch_supervision_edges = []
            
            # train RL and gather data
            iterate = tqdm.tqdm(
                    range(0, train_episodes_per_epoch, num_processes))
            for i in iterate:
                
                # initialize data storage
                batch_observations = []
                batch_hidden_nodes = []
                batch_rewards = []
                batch_action_logps = []
                batch_supervision_edges = []
                finished = torch.zeros(num_processes, dtype=torch.bool)
                class_progress = torch.zeros(
                        num_processes, max_instances, dtype=torch.long).cuda()
                brick_vector_progress = torch.zeros(
                        num_processes, max_instances, 512).cuda()
                
                # reset environment
                observations = train_multi_env.call_method('reset')
                batch_observations.append(observations)
                hidden_nodes = train_multi_env.call_method('get_hidden_nodes')
                batch_hidden_nodes.append(hidden_nodes)
                
                # get epoch-level supervision data
                node_and_edge_labels = train_multi_env.call_method(
                        'get_node_and_edge_labels')
                batch_node_labels, batch_edge_labels = zip(
                        *node_and_edge_labels)
                epoch_node_labels.extend(batch_node_labels)
                epoch_edge_labels.extend(batch_edge_labels)
                
                # rollout
                while not torch.all(finished):
                    x = torch.stack(
                            tuple(torch.stack(
                            tuple(to_tensor(o) for o in obs))
                            for obs in observations)).cuda()
                    (brick_features,
                     class_logits,
                     confidence_logits,
                     action_logits) = node_model(x)
                    
                    action_distribution = torch.distributions.Categorical(
                            logits = action_logits)
                    actions = action_distribution.sample()
                    batch_action_logps.append(
                            action_distribution.log_prob(actions))
                    
                    confidence = torch.sigmoid(confidence_logits)
                    class_prediction = torch.argmax(class_logits, dim=-1)
                    overwrite = (
                            (confidence > confidence_threshold) *
                            class_prediction != 0)
                    class_progress = (
                            class_progress * (~overwrite) +
                            class_prediction * overwrite).detach()
                    brick_vector_progress = (
                            brick_vector_progress * (~overwrite.unsqueeze(-1)) +
                            brick_features * overwrite.unsqueeze(-1)).detach()
                    
                    edge_progress = edge_model(brick_vector_progress).detach()
                    
                    batch_actions = []
                    for j in range(num_processes):
                        batch_actions.append({'action' : {
                                'hide' : int(actions[j]),
                                'node_class' :
                                    class_progress[j].detach().cpu().numpy(),
                                'edge_matrix' :
                                    edge_progress[j].detach().cpu().numpy()}})
                    
                    observations, rewards, terminal, _ = zip(
                            *train_multi_env.call_method('step', batch_actions))
                    hidden_nodes = train_multi_env.call_method(
                            'get_hidden_nodes')
                    batch_hidden_nodes.append(hidden_nodes)
                    batch_observations.append(observations)
                    batch_rewards.append(rewards)
                    terminal = torch.BoolTensor(terminal)
                    finished |= terminal
                
                # REINFORCE
                returns = []
                prev_rewards = [0] * num_processes
                for rewards in batch_rewards[::-1]:
                    returns.append([
                            r+discount*p
                            for r,p in zip(rewards, prev_rewards)])
                    prev_rewards = rewards
                returns = torch.FloatTensor(list(reversed(returns)))
                returns = (returns - returns.mean()) / (returns.std() + EPS)
                returns = returns.cuda()
                batch_action_logps = torch.stack(batch_action_logps)
                loss = torch.mean(-batch_action_logps * returns)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                iterate.set_description('Loss: %.04f'%float(loss))
                
                # Epoch level data
                epoch_observations.extend(zip(*batch_observations))
                epoch_hidden_nodes.extend(zip(*batch_hidden_nodes))
            
            # train supervised
            for mini_epoch in range(1, mini_epochs+1):
                print('Mini Epoch: %i'%mini_epoch)
                num_episodes = len(epoch_observations)
                episode_order = torch.randperm(num_episodes)
                iterate = tqdm.tqdm(episode_order)
                running_class_loss = None
                running_confidence_loss = None
                running_edge_loss = None
                for episode_id in iterate:
                    episode_observations = epoch_observations[episode_id]
                    episode_hidden_nodes = epoch_hidden_nodes[episode_id]
                    episode_node_labels = epoch_node_labels[episode_id]
                    episode_edge_labels = epoch_edge_labels[episode_id]
                    x = torch.stack(
                            tuple(torch.stack(
                            tuple(to_tensor(o) for o in step_observations))
                            for step_observations in episode_observations))
                    
                    # node forward pass
                    (brick_features,
                     class_logits,
                     confidence_logits,
                     action_logits) = node_model(x.cuda())
                    num_steps, num_images, num_instances = class_logits.shape
                    
                    # edge forward pass
                    flat_brick_features = brick_features.view(
                            1, num_steps * num_images, -1)
                    edge_logits = edge_model(flat_brick_features)
                    
                    # class loss
                    class_targets = torch.zeros(
                            num_steps, num_images, dtype=torch.long)
                    for node, label in episode_node_labels.items():
                        node_id = int(node)-1
                        class_targets[:,node_id] = label
                    for i, hidden_nodes in enumerate(episode_hidden_nodes):
                        for hidden_node in hidden_nodes:
                            node_id = int(hidden_node)-1
                            class_targets[i, node_id] = 0
                    class_targets = class_targets.cuda()
                    
                    class_logits = class_logits.view(num_steps * num_images, -1)
                    class_targets = class_targets.view(num_steps * num_images)
                    class_loss = torch.nn.functional.cross_entropy(
                            class_logits, class_targets)
                    if running_class_loss is None:
                        running_class_loss = float(class_loss)
                    else:
                        running_class_loss = (
                                running_class_loss * 0.75 +
                                float(class_loss) * 0.25)
                    
                    # confidence loss
                    class_prediction = torch.argmax(class_logits, dim=-1)
                    confidence_target = (class_prediction == class_targets)
                    confidence_logits = confidence_logits.view(
                            num_steps * num_images)
                    confidence = torch.sigmoid(confidence_logits)
                    confidence_loss = torch.nn.functional.binary_cross_entropy(
                            confidence, confidence_target.float())
                    if running_confidence_loss is None:
                        running_confidence_loss = float(confidence_loss)
                    else:
                        running_confidence_loss = (
                                running_confidence_loss * 0.75 +
                                float(confidence_loss) * 0.25)
                    
                    # edge loss
                    edge_targets = torch.zeros(edge_logits.shape)
                    edge_weights = torch.ones(edge_logits.shape) * 0.05
                    episode_connection_labels = {
                            (a,b) for a,b,c,d in episode_edge_labels}
                    #print(episode_connection_labels)
                    #print(episode_hidden_nodes)
                    #print('------')
                    for im_a in range(num_images):
                        for im_b in range(im_a+1, num_images):
                            if (im_a+1,im_b+1) not in episode_connection_labels:
                                continue
                            for st_a in range(num_steps):
                                if im_a+1 in episode_hidden_nodes[st_a]:
                                    break
                                ind_a = st_a * num_images + im_a
                                if (confidence[st_a * num_images + im_a] <
                                        confidence_threshold):
                                    edge_weights[0, ind_a, :] = 0.0
                                    edge_weights[0, :, ind_a] = 0.0
                                    continue
                                for st_b in range(num_steps):
                                    ind_b = st_b * num_images + im_b
                                    if im_b+1 in episode_hidden_nodes[st_b]:
                                        break
                                    if (confidence[st_b * num_images + im_b] <
                                            confidence_threshold):
                                        edge_weights[0, ind_b, :] = 0.0
                                        edge_weights[0, :, ind_b] = 0.0
                                        continue
                                    #print(ind_a, ind_b)
                                    edge_targets[0, ind_a, ind_b] = 1.0
                                    edge_targets[0, ind_b, ind_a] = 1.0
                                    edge_weights[0, ind_a, ind_b] = 1.0
                                    edge_weights[0, ind_b, ind_a] = 1.0
                    #print(torch.sum(edge_targets))
                    edge_targets = edge_targets.cuda()
                    edge_weights = edge_weights.cuda()
                    edge_loss = torch.nn.functional.binary_cross_entropy(
                            torch.sigmoid(edge_logits),
                            edge_targets,
                            weight = edge_weights) * 0.
                    if running_edge_loss is None:
                        running_edge_loss = float(edge_loss)
                    else:
                        running_edge_loss = (running_edge_loss * 0.75 +
                            float(edge_loss) * 0.25)
                    
                    # combine losses
                    loss = class_loss + confidence_loss
                    loss.backward()
                    
                    optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    iterate.set_description(
                            'cls: %.04f conf: %.04f edg: %.04f'%(
                                running_class_loss,
                                running_confidence_loss,
                                running_edge_loss))
            
            # checkpoint
            if epoch % checkpoint_frequency == 0:
                checkpoint_path = './checkpoint_%04i.pt'%epoch
                print('Saving checkpoint to: %s'%checkpoint_path)
                torch.save((
                        node_model.state_dict(),
                        edge_model.state_dict(),
                        optimizer.state_dict()),
                        checkpoint_path)
            
            # test RL
            if epoch % test_frequency == 0:
                print('Test')
                with torch.no_grad():
                    iterate = tqdm.tqdm(
                            range(0, test_episodes_per_epoch, num_processes))
                    step_rewards = []
                    for i in iterate:
                        
                        # initialize data storage
                        finished = torch.zeros(num_processes, dtype=torch.bool)
                        class_progress = torch.zeros(
                                num_processes,
                                max_instances,
                                dtype=torch.long).cuda()
                        brick_vector_progress = torch.zeros(
                                num_processes, max_instances, 512).cuda()
                        
                        # reset environment
                        observations = test_multi_env.call_method('reset')
                        if i == 0:
                            num_images = len(observations[0])
                            for j in range(num_images):
                                Image.fromarray(observations[0][j]).save(
                                        './image_%04i_0_0_%02i.png'%(epoch, j))
                        hidden_nodes = test_multi_env.call_method(
                                'get_hidden_nodes')
                        
                        # rollout
                        step = 0
                        while not torch.all(finished):
                            x = torch.stack(
                                    tuple(torch.stack(
                                    tuple(to_tensor(o) for o in obs))
                                    for obs in observations)).cuda()
                            (brick_features,
                             class_logits,
                             confidence_logits,
                             action_logits) = node_model(x)
                            
                            actions = torch.argmax(action_logits, dim=-1)
                            '''
                            if i == 0:
                                print('-----')
                                print('Step: %i'%step)
                                print('Action: %i'%actions[0])
                            '''
                            #batch_action_logps.append(
                            #        action_distribution.log_prob(actions))
                            
                            confidence = torch.sigmoid(confidence_logits)
                            class_prediction = torch.argmax(
                                    class_logits, dim=-1)
                            overwrite = (
                                    (confidence > confidence_threshold) *
                                    class_prediction != 0)
                            class_progress = (
                                    class_progress * (~overwrite) +
                                    class_prediction * overwrite).detach()
                            brick_vector_progress = (
                                    brick_vector_progress * (
                                        ~overwrite.unsqueeze(-1)) +
                                    brick_features *
                                        overwrite.unsqueeze(-1)).detach()
                            '''
                            if i == 0:
                                print('Class prediction')
                                print(class_prediction)
                                print('Overwrite')
                                print(overwrite)
                            '''
                            edge_progress = edge_model(
                                    brick_vector_progress).detach()
                            
                            batch_actions = []
                            for j in range(num_test_processes):
                                batch_actions.append({'action' : {
                                    'hide' : int(actions[j]),
                                    'node_class' :
                                      class_progress[j].detach().cpu().numpy(),
                                    'edge_matrix' :
                                      edge_progress[j].detach().cpu().numpy()}})
                            
                            observations, rewards, terminal, _ = zip(
                                    *test_multi_env.call_method(
                                        'step', batch_actions))
                            if i == 0:
                                #print('Reward: %.04f'%rewards[0])
                                num_images = len(observations[0])
                                for j in range(num_images):
                                    Image.fromarray(observations[0][j]).save(
                                            './image_%04i_0_%02i_%02i.png'%(
                                            epoch, step+1, j))
                            #total_reward += sum(rewards)
                            #total_steps += len(rewards)
                            if step >= len(step_rewards):
                                step_rewards.append(0)
                            step_rewards[step] += sum(rewards)
                            
                            hidden_nodes = test_multi_env.call_method(
                                    'get_hidden_nodes')
                            terminal = torch.BoolTensor(terminal)
                            finished |= terminal
                            step += 1
                        
                        #final_reward += sum(rewards)
                        
                        #iterate.set_description('Loss: %.04f'%float(loss))
                    #print('Average Reward: %.04f'%(total_reward/total_steps))
                    #print('Average Final Reward: %.04f'%(
                    #        final_reward/test_episodes_per_epoch))
                    for step, reward in enumerate(step_rewards):
                        print('Average Reward at step %02i: %.04f'%(
                                step, reward/test_episodes_per_epoch))

def supervise_most_confident(
        dataset):
    pass
    
