import time
import collections

import torch

from torch_sparse import coalesce
from torch_scatter import scatter_max, scatter_add

from torch_geometric.data import Data as GraphData
from torch_geometric.utils import to_dense_adj

class BrickList(GraphData):
    '''
    This is an intentionally limited torch_geometric graph with no edges or
    faces allowed.  It's constructor has been modified to only accept keyword
    arguments, all of which must be brick-centric feature names.  These names
    are stored on the instance for future use when merging into a BrickGraph.
    '''
    def __init__(self, brick_feature_spec = None, **kwargs):
        if brick_feature_spec is not None:
            assert len(kwargs) == 0
            kwargs = {}
            for feature_name, spec in brick_feature_spec.items():
                channels, dtype = spec
                kwargs[feature_name] = torch.zeros(
                        (0, channels), dtype=dtype)
            num_nodes = 0
        
        brick_feature_names = tuple(sorted(kwargs.keys()))
        assert len(brick_feature_names) != 0
        num_nodes = kwargs[brick_feature_names[0]].shape[0]
        assert all(kwargs[feature_name].shape[0] == num_nodes
                for feature_name in brick_feature_names[1:])
        
        super(BrickList, self).__init__(num_nodes=num_nodes, **kwargs)
        assert self.edge_index is None
        assert self.edge_attr is None
        assert self.face is None
        self.brick_feature_names = brick_feature_names
    
    def brick_feature_spec(self):
        return {feature_name : (
                    self[feature_name].shape[1], self[feature_name].dtype)
                for feature_name in self.brick_feature_names}
    
    '''
    def num_bricks(self):
        if not len(self.brick_feature_names):
            return 0
        else:
            return self[self.brick_feature_names[0]].shape[0]
    '''
    
    @staticmethod
    def segmentations_to_brick_lists_average(
            score_logits, segmentation, feature_dict, max_instances=None):
        b, h, w = (
                score_logits.shape[0],
                score_logits.shape[-2],
                score_logits.shape[-1])
        score_logits = score_logits.view(b,1,h,w)
        
        scores = torch.sigmoid(score_logits)
        total_scores = scatter_add(
                scores.view(b,1,-1), segmentation.view(b,1,-1))
        
        # there is probably a more efficient way to do this
        normalizer = scatter_add(
                torch.ones_like(scores.view(b,1,-1)),
                segmentation.view(b,1,-1))
        
        segment_features = {}
        for feature_name, feature_values in feature_dict.items():
            c = feature_values.shape[1]
            weighted_feature_values = (scores * feature_values).view(b, c, -1)
            segment_features[feature_name] = scatter_add(
                    weighted_feature_values, segmentation.view(b, 1, -1))
        
        '''
        if (max_instances is not None and
                max_instances < raw_segment_score.shape[1]):
            raw_segment_score, remap_segment_id = torch.topk(
                    raw_segment_score, max_instances)
            raw_source_index = torch.gather(
                    raw_source_index, 1, remap_segment_id)
        else:
            remap_segment_id = None
        '''
        
        valid_segments = torch.where(normalizer > 0.)
        
        brick_lists = []
        for i in range(b):
            item_entries = valid_segments[0] == i
            nonzero_ids = valid_segments[-1][item_entries]
            num_segments = nonzero_ids.shape[0]
            batch_entries = [i] * num_segments
            
            segment_total_scores = total_scores[batch_entries, :, nonzero_ids]
            segment_normalizer = normalizer[batch_entries, :, nonzero_ids]
            segment_normalized_scores = segment_total_scores/segment_normalizer
            
            if (max_instances is not None and
                    max_instances < segment_total_scores.shape[0]):
                
                '''
                # this is wrong but performs better for some reason
                segment_total_scores, topk = torch.topk(
                        segment_total_scores.view(-1), max_instances)
                segment_total_scores = segment_total_scores.view(-1,1)
                segment_normalized_scores = segment_normalized_scores[topk]
                '''
                segment_normalized_scores, topk = torch.topk(
                        segment_normalized_scores.view(-1), max_instances)
                segment_normalized_scores = segment_normalized_scores.view(-1,1)
                segment_total_scores = segment_total_scores[topk]
                
                segment_ids = nonzero_ids[topk]
            else:
                topk = None
                segment_ids = nonzero_ids
            
            graph_data = {}
            for feature_name, feature_values in segment_features.items():
                features = feature_values[batch_entries, :, nonzero_ids]
                if topk is not None:
                    features = features[topk]
                features = features / segment_total_scores
                graph_data[feature_name] = features
            
            brick_lists.append(BrickList(
                    score=segment_normalized_scores,
                    segment_id=segment_ids.unsqueeze(1),
                    **graph_data))
        
        batch = BrickListBatch(brick_lists)
        
        return batch
    
    @staticmethod
    def segmentations_to_brick_lists(
            score_logits, segmentation, feature_dict, max_instances=None):
        # get dimensions (this works for either 3 or 4 dimensional tensors)
        b, h, w = (
                score_logits.shape[0],
                score_logits.shape[-2],
                score_logits.shape[-1])
        score_logits = score_logits.view(b,h,w)
        
        # scatter_max the score_logits using the segmentation
        # raw_segment_score: the highest score for each segment
        # raw_source_index: the pixel location where raw_segment_score was found
        score = torch.sigmoid(score_logits)
        raw_segment_score, raw_source_index = scatter_max(
                score.view(b,-1), segmentation.view(b,-1))
        
        if (max_instances is not None and
                max_instances < raw_segment_score.shape[1]):
            raw_segment_score, remap_segment_id = torch.topk(
                    raw_segment_score, max_instances)
            raw_source_index = torch.gather(
                    raw_source_index, 1, remap_segment_id)
        else:
            remap_segment_id = None
        
        # filter out the segments that have no contributing pixels
        # segment_score: filtered raw_segment_score
        # source_index: filtered raw_source_index
        valid_segments = torch.where(raw_segment_score > 0.)
        
        brick_lists = []
        for i in range(b):
            item_entries = valid_segments[0] == i
            nonzero_ids = valid_segments[1][item_entries]
            if remap_segment_id is None:
                segment_ids = nonzero_ids
            else:
                segment_ids = remap_segment_id[i][nonzero_ids]
            
            num_segments = nonzero_ids.shape[0]
            batch_entries = [i] * num_segments
            
            item_segment_score = raw_segment_score[batch_entries, nonzero_ids]
            item_source_index = raw_source_index[batch_entries, nonzero_ids]
            
            positions = torch.stack(
                    (item_source_index // w, item_source_index % w), dim=1)
            
            graph_data = {}
            for feature_name, feature_values in feature_dict.items():
                c = feature_values.shape[1]
                segment_features = feature_values.view(b, c, -1)[
                        batch_entries, :, item_source_index]
                graph_data[feature_name] = segment_features

            brick_lists.append(BrickList(
                    score=item_segment_score.unsqueeze(1),
                    segment_id=segment_ids.unsqueeze(1),
                    pos=positions,
                    **graph_data))
        
        batch = BrickListBatch(brick_lists)
        
        return batch
    
    @staticmethod
    def single_best_brick_lists(
            score_logits, feature_dict):
        # get dimensions (this works for either 3 or 4 dimensional tensors)
        b, h, w = (
                score_logits.shape[0],
                score_logits.shape[-2],
                score_logits.shape[-1])
        score_logits = score_logits.view(b,h*w)
        
        '''
        # scatter_max the score_logits using the segmentation
        # raw_segment_score: the highest score for each segment
        # raw_source_index: the pixel location where raw_segment_score was found
        score = torch.sigmoid(score_logits)
        raw_segment_score, raw_source_index = scatter_max(
                score.view(b,-1), segmentation.view(b,-1))
        
        if (max_instances is not None and
                max_instances < raw_segment_score.shape[1]):
            raw_segment_score, remap_segment_id = torch.topk(
                    raw_segment_score, max_instances)
            raw_source_index = torch.gather(
                    raw_source_index, 1, remap_segment_id)
        else:
            remap_segment_id = None
        
        # filter out the segments that have no contributing pixels
        # segment_score: filtered raw_segment_score
        # source_index: filtered raw_source_index
        valid_segments = torch.where(raw_segment_score > 0.)
        '''
        
        max_score, max_score_locations = torch.max(score_logits, dim=-1)
        positions = torch.stack(
                (max_score_locations // w, max_score_locations % w), dim=1)
        
        brick_lists = []
        for i in range(b):
            '''
            item_entries = valid_segments[0] == i
            nonzero_ids = valid_segments[1][item_entries]
            if remap_segment_id is None:
                segment_ids = nonzero_ids
            else:
                segment_ids = remap_segment_id[i][nonzero_ids]
            
            num_segments = nonzero_ids.shape[0]
            batch_entries = [i] * num_segments
            
            item_segment_score = raw_segment_score[batch_entries, nonzero_ids]
            item_source_index = raw_source_index[batch_entries, nonzero_ids]
            
            positions = torch.stack(
                    (item_source_index // w, item_source_index % w), dim=1)
            '''
            graph_data = {}
            for feature_name, feature_values in feature_dict.items():
                c = feature_values.shape[1]
                #segment_features = feature_values.view(b, c, -1)[
                #        batch_entries, :, item_source_index]
                segment_features = feature_values.view(b, c, -1)[
                         [i], :, max_score_locations[i]]
                graph_data[feature_name] = segment_features
            
            brick_lists.append(BrickList(
                    score=torch.sigmoid(max_score[[i]]).unsqueeze(1),
                    #score=item_segment_score.unsqueeze(1),
                    #segment_id=segment_ids.unsqueeze(1),
                    #pos=positions,
                    pos=positions[[i]],
                    **graph_data))
        
        batch = BrickListBatch(brick_lists)
        
        return batch
    
    def cuda(self):
        features = {feature_name : self[feature_name].cuda()
                for feature_name in self.brick_feature_names}
        return BrickList(**features)
        '''
        for feature_name in self.brick_feature_names:
            self[feature_name] = self[feature_name].cuda()
        return self
        '''
    
    def cpu(self):
        features = {feature_name : self[feature_name].cpu()
                for feature_name in self.brick_feature_names}
        return BrickList(**features)
        '''
        for feature_name in self.brick_feature_names:
            self[feature_name] = self[feature_name].cpu()
        return self
        '''
    
    def to(self, device):
        features = {feature_name : self[feature_name].to(device)
                for feature_name in self.brick_feature_names}
        return BrickList(**features)
        '''
        for feature_name in self.brick_feature_names:
            self[feature_name] = self[feature_name].to(device)
        return self
        '''
    
    def detach(self):
        features = {feature_name : self[feature_name].detach()
                for feature_name in self.brick_feature_names}
        return BrickList(**features)

class BrickGraph(GraphData):
    def __init__(self,
            brick_list=None,
            edge_index=None,
            edge_attr=None,
            edge_attr_channels=None):
        
        # error assertions
        assert isinstance(brick_list, BrickList)
        
        # if edge_index or edge_attr are None, replace them with empty tensors
        if edge_index is None:
            device = brick_list[brick_list.brick_feature_names[0]].device
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        if edge_attr is None:
            if edge_attr_channels is None:
                edge_attr_channels = 0
            edge_attr = torch.zeros(
                    edge_index.shape[1],
                    edge_attr_channels,
                    device=edge_index.device)
        assert edge_index.device == edge_attr.device
        
        # super initialize
        super(BrickGraph, self).__init__(
                num_nodes=brick_list.num_nodes,
                edge_index=edge_index,
                edge_attr=edge_attr)
        assert self.face is None
        
        # pull features from brick_list
        for feature_name in brick_list.brick_feature_names:
            self[feature_name] = brick_list[feature_name]
            assert self.edge_index.device == self[feature_name].device
        self.brick_feature_names = brick_list.brick_feature_names
    
    def brick_feature_spec(self):
        return {feature_name : (
                    self[feature_name].shape[1], self[feature_name].dtype)
                for feature_name in self.brick_feature_names}
    
    def merge(self,
            other,
            matching_nodes=None,
            new_edges=None,
            new_edge_attr=None):
        
        assert isinstance(other, BrickGraph)
        assert self.brick_feature_names == other.brick_feature_names
        assert self.edge_attr.shape[1] == other.edge_attr.shape[1]
        
        merged_brick_features = {feature_name : self[feature_name].clone()
                for feature_name in self.brick_feature_names}
        
        device = self.edge_index.device
        assert other.edge_index.device == device
        if matching_nodes is None:
            append_indices = torch.ones(other.num_nodes, dtype=torch.bool)
            destination = torch.arange(
                    self.num_nodes,
                    self.num_nodes + other.num_nodes).to(device)
        else:
            other_indices, self_indices = matching_nodes
            for feature_name in self.brick_feature_names:
                merged_brick_features[feature_name][self_indices] = (
                        other[feature_name][other_indices])
            append_indices = torch.ones(other.num_nodes, dtype=torch.bool)
            append_indices[other_indices] = False
            destination = torch.zeros(other.num_nodes, dtype=torch.long).to(
                    device)
            destination[other_indices] = self_indices
            append_size = other.num_nodes - other_indices.shape[0]
            destination[append_indices] = torch.arange(
                    self.num_nodes,
                    self.num_nodes + append_size).to(device)
        
        for feature_name in self.brick_feature_names:
            merged_brick_features[feature_name] = torch.cat(
                    (merged_brick_features[feature_name],
                     other[feature_name][append_indices]))
        
        # remap other's edges
        remapped_other_edges = destination[other.edge_index]
        new_edge_index_tensors = [self.edge_index, remapped_other_edges]
        new_edge_attr_tensors = [self.edge_attr, other.edge_attr]
        
        # add new (bidirectional) edges
        if new_edges is not None:
            assert new_edges.device == self.edge_index.device
            other_nodes, self_nodes = new_edges
            remapped_other_nodes = destination[other_nodes]
            remapped_new_edges = torch.stack((
                    torch.cat((remapped_other_nodes, self_nodes)),
                    torch.cat((self_nodes, remapped_other_nodes))))
            new_edge_index_tensors.append(remapped_new_edges)
            
            if new_edge_attr is None:
                new_edge_attr = torch.zeros(
                        new_edges.shape[1], 0, device=new_edges.device)
            assert new_edge_attr.device == new_edges.device
            bidirectional_new_edge_attr = torch.cat(
                    (new_edge_attr, new_edge_attr), dim=0)
            new_edge_attr_tensors.append(bidirectional_new_edge_attr)
        
        merged_edge_index = torch.cat(new_edge_index_tensors, dim=1)
        merged_edge_attr = torch.cat(new_edge_attr_tensors, dim=0)
        
        merged_brick_list = BrickList(**merged_brick_features)
        merged_graph = BrickGraph(
                merged_brick_list, merged_edge_index, merged_edge_attr)
        
        if new_edges is not None:
            #merged_graph.coalesce()
            # do this explicitly with torch_sparse.coalesce in order to be
            # able to use the 'max' operation
            (merged_graph.edge_index,
             merged_graph.edge_attr) = coalesce(
                    merged_graph.edge_index,
                    merged_graph.edge_attr,
                    merged_graph.num_nodes,
                    merged_graph.num_nodes,
                    op='max')
        
        return merged_graph
    
    def num_edges(self):
        return self.edge_index.shape[1]
    
    def edge_dict(self, tag=None):
        '''
        Returns a dictionary of the form:
        {
            (tag, i, j, label_i, label_j) : score
        }
        '''
        assert 'instance_label' in self.brick_feature_names
        assert self.edge_attr.shape[1] >= 1
        edge_dict = {}
        if not self.num_nodes or not self.num_edges:
            return edge_dict
        
        instance_labels = torch.argmax(self.instance_label, dim=1).cpu()
        for i in range(self.num_edges()):
            key = []
            if tag is not None:
                key.append(tag)
            
            node_a, node_b = self.edge_index[:,i].cpu()
            if node_a > node_b:
                node_a, node_b = node_b, node_a
            node_a, node_b = int(node_a), int(node_b)
            key.extend([node_a, node_b])
            
            label_a = int(instance_labels[node_a])
            label_b = int(instance_labels[node_b])
            key.extend([label_a, label_b])
            
            key = tuple(key)
            edge_dict[key] = float(self.edge_attr[i,0].cpu())
        
        return edge_dict
    
    def edge_matrix(self, bidirectionalize=False):
        edge_index = self.edge_index
        if bidirectionalize:
            flipped_edge_index = torch.cat((edge_index[[1]], edge_index[[0]]))
            edge_index = torch.cat((edge_index, flipped_edge_index), dim=1)
        matrix = to_dense_adj(edge_index, max_num_nodes=self.num_nodes)
        return matrix[0]
    
    def cuda(self):
        brick_list = BrickList(**{feature_name : self[feature_name]
                for feature_name in self.brick_feature_names}).cuda()
        edge_index = self.edge_index.cuda()
        edge_attr = self.edge_attr.cuda()
        return BrickGraph(
                brick_list, edge_index=edge_index, edge_attr=edge_attr)
        '''
        for feature_name in self.brick_feature_names:
            self[feature_name] = self[feature_name].cuda()
        self.edge_index = self.edge_index.cuda()
        self.edge_attr = self.edge_attr.cuda()
        return self
        '''
    
    def cpu(self):
        brick_list = BrickList(**{feature_name : self[feature_name]
                for feature_name in self.brick_feature_names}).cpu()
        edge_index = self.edge_index.cpu()
        edge_attr = self.edge_attr.cpu()
        return BrickGraph(
                brick_list, edge_index=edge_index, edge_attr=edge_attr)
        
        '''
        for feature_name in self.brick_feature_names:
            self[feature_name] = self[feature_name].cpu()
        self.edge_index = self.edge_index.cpu()
        self.edge_attr = self.edge_attr.cpu()
        return self
        '''
    
    def to(self, device):
        brick_list = BrickList(**{feature_name : self[feature_name]
                for feature_name in self.brick_feature_names}).to(device)
        edge_index = self.edge_index.to(device)
        edge_attr = self.edge_attr.to(device)
        return BrickGraph(
                brick_list, edge_index=edge_index, edge_attr=edge_attr)
        '''
        for feature_name in self.brick_feature_names:
            self[feature_name] = self[feature_name].to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        return self
        '''
    
    def detach(self):
        brick_list = BrickList(**{feature_name : self[feature_name]
                for feature_name in self.brick_feature_names}).detach()
        edge_index = self.edge_index.detach()
        edge_attr = self.edge_attr.detach()
        return BrickGraph(
                brick_list, edge_index=edge_index, edge_attr=edge_attr)

class ClassTypeBatch(collections.abc.MutableSequence):
    ClassType = None
    def __init__(self, class_type_list):
        assert all(isinstance(class_type_list, self.ClassType)
                for class_type_list in class_type_list)
        self.class_type_list = list(class_type_list)
    
    def __len__(self):
        return len(self.class_type_list)
    
    def __getitem__(self, index):
        try:
            _ = len(index)
            return type(self)(list(self.class_type_list[i] for i in index))
        except TypeError:
            return self.class_type_list[index]
    
    def __setitem__(self, index, value):
        assert isinstance(value, self.ClassType)
        self.class_type_list[index] = value
    
    def __delitem__(self, index):
        del(self.class_type_list[index])
    
    def insert(self, index, value):
        assert isinstance(value, self.ClassType)
        self.class_type_list.insert(index, value)
    
    def __add__(self, other):
        assert isinstance(other, type(self))
        return type(self)(self.class_type_list + other.class_type_list)
    
    @classmethod
    def join(cls, others, transpose=False):
        if transpose:
            return cls([other[i] for i in range(len(others[0]))
                    for other in others])
        else:
            return cls(sum((other.class_type_list for other in others), []))
    
    # device/detach
    def cuda(self):
        cuda_class_type_list = [c.cuda() for c in self.class_type_list]
        return type(self)(cuda_class_type_list)
        #for c in self.class_type_list:
        #    c.cuda()
        #return self
    
    def cpu(self):
        cpu_class_type_list = [c.cpu() for c in self.class_type_list]
        return type(self)(cpu_class_type_list)
        #for c in self.class_type_list:
        #    c.cpu()
        #return self
    
    def to(self, device):
        device_class_type_list = [c.to(device) for c in self.class_type_list]
        return type(self)(device_class_type_list)
        #for c in self.class_type_list:
        #    c.to(device)
        #return self
    
    def detach(self):
        detached_class_type_list = [c.detach() for c in self.class_type_list]
        return type(self)(detached_class_type_list)

class BrickListBatch(ClassTypeBatch):
    ClassType = BrickList

class BrickGraphBatch(ClassTypeBatch):
    ClassType = BrickGraph
    
    @staticmethod
    def from_brick_list_batch(brick_list_batch, edge_index):
        brick_graphs = [BrickGraph(brick_list, edge_index[i])
                for i, brick_list in enumerate(brick_list_batch)]
        return BrickGraphBatch(brick_graphs)
    
    def edge_matrix(self):
        return torch.stack(
                tuple(graph.edge_matrix() for graph in self.class_type_list))
