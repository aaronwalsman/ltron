import torch
from torchvision.transforms.functional import to_tensor

import numpy

import gym.spaces as spaces

from brick_gym.gym.spaces import (
        ImageSpace, SegmentationSpace, StepSpace, InstanceListSpace,
        EdgeSpace, InstanceGraphSpace)
from brick_gym.torch.brick_geometric import (
        BrickList, BrickGraph, BrickListBatch, BrickGraphBatch)

def gym_space_to_tensors(
        data, space, device=torch.device('cpu'), image_transform=to_tensor):
    def recurse(data, space):
        if isinstance(space, ImageSpace):
            if len(data.shape) == 3:
                tensor = image_transform(data)
            elif len(data.shape) == 4:
                tensor = torch.stack(
                        tuple(image_transform(image) for image in data))
            return tensor.to(device)
        elif isinstance(space, SegmentationSpace):
            return torch.LongTensor(data).to(device)
        
        elif isinstance(space, StepSpace):
            return torch.LongTensor(data).to(device)
        
        elif isinstance(space, InstanceListSpace):
            num_instances = data['num_instances']
            labels = data['label']
            if len(labels.shape) == 2:
                #labels = labels[:num_instances]
                tensor = torch.LongTensor(labels).to(device)
                # TODO: SCORE
                return BrickList(instance_label = tensor)
            elif len(data['label'].shape) == 3:
                brick_lists = []
                for n, l in zip(num_instances, labels):
                    #l = l[:n]
                    tensor = torch.LongTensor(l).to(device)
                    # TODO: SCORE
                    brick_lists.append(BrickList(instance_label = tensor))
                return BrickListBatch(brick_lists)
        
        elif isinstance(space, EdgeSpace):
            # TODO: SCORE
            tensor_dict = {
                    'edge_index' :
                        torch.LongTensor(data['edge_index']).to(device)
            }
            return tensor_dict
        
        elif isinstance(space, InstanceGraphSpace):
            brick_list = recurse(data['instances'], space['instances'])
            edge_tensor_dict = recurse(data['edges'], space['edges'])
            edge_index = edge_tensor_dict['edge_index']
            # TODO: SCORE
            if isinstance(brick_list, BrickList):
                return BrickGraph(brick_list, edge_index=edge_index)
            elif isinstance(brick_list, BrickListBatch):
                return BrickGraphBatch.from_brick_list_batch(
                        brick_list, edge_index)
        
        # keep the default spaces last because brick_gym's custom spaces
        # inherit from them so those cases should be caught first
        elif isinstance(space, spaces.Discrete):
            return data
        
        elif isinstance(space, spaces.Dict):
            return {key : recurse(data[key], space[key]) for key in data}
        
        elif isinstance(space, spaces.Tuple):
            return tuple(recurse(d, s) for d,s in zip(data, space))
    
    return recurse(data, space)

def gym_space_list_to_tensors(
        data, space, device=torch.device('cpu'), image_transform=to_tensor):
    '''
    Everything added here should be set up so that if it already has a batch
    dimension, that should become the PRIMARY dimension (dimension 0) so that
    when everything is viewed as one long list, neighboring entries in the
    batch dimension come from the same sequence.
    '''
    tensors = [gym_space_to_tensors(d, space, device, image_transform)
            for d in data]
    
    def recurse(data, space):
        if isinstance(space, ImageSpace):
            c, h, w = data[0].shape[-3:]
            tensor = torch.stack(data, dim=-4)
            return tensor.view(-1, c, h, w)
        
        elif isinstance(space, SegmentationSpace):
            h, w = data[0].shape[-2:]
            tensor = torch.stack(data, dim=-3)
            return tensor.view(-1, h, w)
        
        elif isinstance(space, StepSpace):
            c = data[0].shape[-1]
            tensor = torch.stack(data, dim=-2)
            return tensor.view(-1, c)
        
        elif isinstance(space, InstanceGraphSpace):
            return BrickGraphBatch.join(data, transpose=True)
        
        elif isinstance(space, spaces.Dict):
            return {key : recurse([d[key] for d in data], space[key])
                    for key in data[0]}
        
        elif isinstance(space, spaces.Tuple):
            return tuple(recurse(data[i], space[i]) for i in len(data[0]))
    
    return recurse(tensors, space)

def graph_to_gym_space(data, space):
    
    # build the instance labels
    segment_id = data['segment_id'].view(-1).detach().cpu().numpy()
    instance_labels = numpy.zeros(
            (space['instances']['label'].shape[0]), dtype=numpy.long)
    
    if data.num_nodes:
        # discretize the labels
        discrete_labels = torch.argmax(data['instance_label'], dim=-1)
        discrete_labels = discrete_labels.detach().cpu().numpy()
        
        # remap labels
        instance_labels[segment_id] = discrete_labels
        instance_labels = instance_labels.reshape(-1, 1)
        
        # num_instances
        num_instances = min(data.num_nodes, space.max_instances)
    else:
        num_instances = 0
    
    # compile the instance data
    instance_data = {
            'num_instances' : data.num_nodes,
            'label' : instance_labels,
    }
    if space['instances'].include_score:
        instance_data['score'] = data['score'].cpu().numpy()
    
    # remap edges
    original_edges = data['edge_index'].detach().cpu().numpy()
    remapped_edge_index = segment_id[original_edges]
    num_edges = min(data.num_edges(), space.max_edges)
    edge_data = {
            'num_edges' : num_edges,
            'edge_index' : remapped_edge_index,
    }
    if space['edges'].include_score:
        edge_data['score'] = data['edge_attr'].cpu().numpy()
    
    # compile result
    result = {
            'instances' : instance_data,
            'edges' : edge_data,
    }
    #if 'edge_scores' in space.spaces:
    #    edge_score = data['edge_attr'][:,0].detach().cpu().numpy()
    #    result['edge_scores'] = edge_score
    
    return result
