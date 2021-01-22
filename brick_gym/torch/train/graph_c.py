import torch
from torch.distributions import Categorical

import tqdm

from gym.vector.async_vector_env import AsyncVectorEnv

from brick_gym.visualization.utils import save_gym_data
from brick_gym.dataset.paths import get_dataset_info
from brick_gym.gym.brick_env import async_brick_env
import brick_gym.envs.standard_envs as standard_envs
from brick_gym.torch.utils import images_masks_to_segment_tensor
import brick_gym.torch.models.named_models as named_models

def train_label_confidence(
        num_epochs,
        dataset,
        train_split = 'train_mpd',
        train_subset = None,
        test_split = 'test_mpd',
        test_subset = None,
        num_processes = 16,
        batch_size = 64,
        learning_rate = 3e-4,
        node_loss_weight = 0.8,
        confidence_loss_weight = 0.2,
        confidence_ratio = 1.0,
        train_steps_per_epoch = 4096,
        test_frequency = 1,
        test_steps_per_epoch = 1024,
        checkpoint_frequency = 1):
    
    print('Building the model')
    model = named_models.named_graph_model(
            'first_try',
            backbone_name = 'pretrained_attention_resnet18',
            edge_model_name = 'feature_difference_512',
            node_classes = 7,
            shape=(256,256)).cuda()
    
    print('Building the vector environments')
    train_vector_env = async_brick_env(
            num_processes, standard_envs.graph_env,
            dataset, train_split, train_subset, train=True,
            print_traceback=True)
    test_vector_env = async_brick_env(
            num_processes, standard_envs.graph_env,
            dataset, test_split, test_subset, train=False,
            print_traceback=True)
    dataset_info = get_dataset_info(dataset)
    
    print('Building the optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # train
    for epoch in range(1, num_epochs+1):
        print('='*80)
        print('Training epoch: %i'%epoch)
        
        print('-'*80)
        step_reward = []
        
        model.train()
        observation = train_vector_env.reset()
        terminal = [False] * num_processes
        iterate = tqdm.tqdm(range(train_steps_per_epoch//num_processes))
        for i in iterate:
            #save_gym_data(
            #        observation, train_vector_env.single_observation_space,
            #        './observation_%06i'%step)
            #print('step', observation['episode_step'])
            #print('term', terminal)
            
            # compute torch objects from the observation/terminal values
            x = images_masks_to_segment_tensor(
                    observation['color'],
                    observation['mask'],
                    dataset_info['max_instances_per_scene']).cuda()
            terminal = torch.BoolTensor(terminal).cuda()
            node_targets = torch.LongTensor(
                    observation['graph_label']['nodes']).cuda()
            edge_targets = torch.FloatTensor(
                    observation['graph_label']['edges']).cuda()
            
            # model forward pass
            (step_node_features,
             step_node_logits,
             visibility_logits,
             node_logits,
             edge_logits) = model(x, terminal)
            
            # supervision losses
            loss = 0
            # node loss
            b, n, c = step_node_logits.shape
            visible_segments = torch.sum(x[:,:,-1], dim=(2,3)) > 0.
            #print('vis_seg', visible_segments)
            visible_node_target = node_targets * visible_segments
            node_loss = torch.nn.functional.cross_entropy(
                    step_node_logits.view(b*n, c),
                    visible_node_target.view(b*n))
            loss = loss + node_loss * node_loss_weight
            
            # confidence/visibility loss
            predicted_node_class = torch.argmax(step_node_logits, dim=-1)
            node_correct = predicted_node_class == visible_node_target
            confidence = torch.sigmoid(visibility_logits)
            confidence_loss = torch.nn.functional.binary_cross_entropy(
                    confidence, node_correct.float(), reduction='none')
            num_visible_segments = torch.sum(visible_segments)
            if num_visible_segments:
                confidence_loss = confidence_loss * visible_segments
                confidence_loss = (
                        torch.sum(confidence_loss) / num_visible_segments)
                loss = loss + confidence_loss * confidence_loss_weight
            
            # backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # description
            iterate.set_description('Train loss: %.04f'%float(loss))
            
            # turn forward pass results into actions
            node_actions = torch.argmax(node_logits, dim=-1).cpu().numpy()
            edge_actions = edge_logits.detach().cpu().numpy()
            visibility_prob = (
                    torch.softmax(visibility_logits, dim=-1) *
                    visible_segments) + 1e-5
            visibility_distribution = Categorical(visibility_prob)
            visibility_actions = (
                    visibility_distribution.sample().cpu().numpy())
            #print('vis_act', visibility_actions)
            actions = [{} for _ in range(num_processes)]
            for j, action in enumerate(actions):
                action['graph'] = {
                        'nodes' : node_actions[j],
                        'edges' : edge_actions[j]}
                action['visibility'] = visibility_actions[j]
            
            # step
            previous_step = observation['episode_step']
            observation, reward, terminal, info = train_vector_env.step(actions)
            for step, r in zip(previous_step, reward):
                if step >= len(step_reward):
                    step_reward.append([0,0])
                step_reward[step][0] += r
                step_reward[step][1] += 1
        
        print('Average train reward:')
        for i, (r,t) in enumerate(step_reward):
            print('Step %i: %.04f'%(i,r/t))
        
        if i%checkpoint_frequency == 0:
            print('-'*80)
            model_path = './model_%04i.pt'
            print('Saving model to: %s'%model_path)
            torch.save(model.state_dict(), checkpoint_path)
            
            optimizer_path = './optimizer_%04i.pt'
            print('Saving optimizer to: %s'%optimizer_path)
            torch.save(optimizer.state_dict(), optimizer_path)
        
        if i%test_frequency == 0:
            with torch.no_grad():
                print('-'*80)
                model.eval()
                step_reward = []
                
                observation = test_vector_env.reset()
                terminal = [False] * num_processes
                iterate = tqdm.tqdm(range(test_steps_per_epoch//num_processes))
                iterate.set_description('Test')
                for i in iterate:
                    
                    # compute torch objects from the observation/terminal values
                    x = images_masks_to_segment_tensor(
                            observation['color'],
                            observation['mask'],
                            dataset_info['max_instances_per_scene']).cuda()
                    terminal = torch.BoolTensor(terminal).cuda()
                    
                    # model forward pass
                    (step_node_features,
                     step_node_logits,
                     visibility_logits,
                     node_logits,
                     edge_logits) = model(x, terminal)
                    
                    # turn forward pass results into actions
                    node_actions = torch.argmax(
                            node_logits, dim=-1).cpu().numpy()
                    edge_actions = edge_logits.detach().cpu().numpy()
                    visibility_actions = torch.argmax(
                            visibility_logits, dim=-1).cpu().numpy()
                    actions = [{} for _ in range(num_processes)]
                    for j, action in enumerate(actions):
                        action['graph'] = {
                                'nodes' : node_actions[j],
                                'edges' : edge_actions[j]}
                        action['visibility'] = visibility_actions[j]
                    
                    # step
                    previous_step = observation['episode_step']
                    (observation,
                     reward,
                     terminal,
                     info) = test_vector_env.step(actions)
                    for step, r in zip(previous_step, reward):
                        if step >= len(step_reward):
                            step_reward.append([0,0])
                        step_reward[step][0] += r
                        step_reward[step][1] += 1
            
            print('Average test reward:')
            for i, (r,t) in enumerate(step_reward):
                print('Step %i: %.04f'%(i,r/t))
