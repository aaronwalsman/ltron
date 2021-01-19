import torch

from torch_scatter import scatter_max

from torch_geometric.data import Data as GraphData, Batch as GraphBatch

class BrickList(GraphData):
    '''
    This is an intentionally limited torch_geometric graph with no edges or
    faces allowed.  It's constructor has been modified to only accept keyword
    arguments, all of which must be brick-centric feature names.  These names
    are stored on the instance for future use when merging into a BrickGraph.
    '''
    def __init__(self, **kwargs):
        super(BrickList, self).__init__(**kwargs)
        assert self.edge_index is None
        assert self.edge_attr is None
        assert self.face is None
        self.brick_feature_names = tuple(sorted(kwargs.keys()))
        assert len(self.brick_feature_names) != 0
    
    def num_bricks(self):
        if not len(self.brick_feature_names):
            return 0
        else:
            return self[self.brick_feature_names[0]].shape[0]

class BrickGraph(GraphData):
    def __init__(self, brick_list, edge_index):
        '''
        edge_index cannot be None, if you want no edges, provide an empty tensor
        '''
        super(BrickGraph, self).__init__()
        assert isinstance(brick_list, BrickList)
        assert edge_index is not None
        for feature_name in brick_list.brick_feature_names:
            self[feature_name] = brick_list[feature_name]
        self.edge_index = edge_index
        self.brick_feature_names = brick_list.brick_feature_names
    
    def merge(self, other, matching_nodes=None, new_edges=None):
        assert isinstance(other, BrickGraph)
        assert self.brick_feature_names == other.brick_feature_names
        if matching_nodes is None:
            append_indices = torch.ones(other.num_bricks(), dtype=torch.bool)
            destination = torch.arange(
                    self.num_bricks(),
                    self.num_bricks() + other.num_bricks()).long()
        else:
            other_indices, self_indices = matching_nodes
            for feature_name in self.brick_feature_names:
                self[feature_name][self_indices] = (
                        other[feature_name][other_indices])
            append_indices = torch.ones(other.num_bricks(), dtype=torch.bool)
            append_indices[other_indices] = False
            destination = torch.zeros(other.num_bricks(), dtype=torch.long)
            destination[other_indices] = self_indices
            append_size = other.num_bricks() - other_indices.shape[0]
            destination[append_indices] = torch.arange(
                    self.num_bricks(),
                    self.num_bricks() + append_size).long()
        
        for feature_name in self.brick_feature_names:
            self[feature_name] = torch.cat(
                    (self[feature_name], other[feature_name][append_indices]))
        
        # remap other's edges
        remapped_other_edges = destination[other.edge_index]
        new_edge_tensors = [self.edge_index, remapped_other_edges]
        
        # add new (bidirectional) edges
        if new_edges is not None:
            other_nodes, self_nodes = new_edges
            remapped_other_nodes = destination[other_nodes]
            remapped_new_edges = torch.stack(
                    torch.cat(remapped_other_nodes, self_nodes),
                    torch.cat(self_nodes, remapped_other_nodes))
            new_edge_tensors.append(remapped_new_edges)
        
        self.edge_index = torch.cat(new_edge_tensors, dim=1)
    
    def num_bricks(self):
        if not len(self.brick_feature_names):
            return 0
        else:
            return self[self.brick_feature_names[0]].shape[0]
    
    #def merge_brick_list(self,
    #        brick_list,
    #        insert_nodes=None,
    #        append_nodes=None,
    #        new_new_edges=None,
    #        old_new_edges=None):
    #    '''
    #    index:     0 1 2 3 4 5 6
    #    old feat:  a b c d e f g
    #    new feat:  h i j k
    #    merge:     7 8 1 3
    #    
    #    result:    a j c k e f g h i
    #    Note that any merge indices that are above the length of what's already
    #    here will be added to the end in the order they exist in brick_list
    #    regardless of the value of merge.
    #    '''
    #    # if no brick list has been added yet, simply add the brick_list as
    #    # nodes and new_new_features as edges
    #    if not self.initialized:
    #        for brick_feature_name in brick_list.brick_feature_names:
    #            self[brick_feature_name] = brick_list[brick_feature_name]
    #        self.edge_index = new_new_edges
    #        self.brick_feature_names = brick_list.brick_feature_names
    #        self.initialized = True
    #    
    #    else:
    #        # check compatibility
    #        assert self.brick_feature_names == brick_list.brick_feature_names
    #        
    #        first_name = self.brick_feature_names[0]
    #        num_old_nodes = int(self[first_name].shape[0])
    #        num_new_nodes = int(brick_list[first_name].shape[0])
    #        
    #        if insert_nodes is not None:
    #            insert_source, insert_target = insert_nodes
    #            for feature_name in brick_list.brick_feature_names:
    #                
    #        
    #        # if no merge vector was specified, concatenate the new nodes onto
    #        # the end of this graph
    #        if merge_nodes is None:
    #            for feature_name in brick_list.brick_feature_names:
    #                self[feature_name] = torch.cat(
    #                        (self[feature_name], brick_list[feature_name]))
    #            if new_new_edges is not None:
    #                self.edge_index = torch.cat(
    #                        (self.edge_index, new_new_edges + num_old_nodes))
    #        
    #        # if a merge vector was specified, merge and append as necessary
    #        else:
    #            # first figure out which will be appended and which inserted
    #            insert_indices = merge_nodes < num_old_nodes
    #            append_indices = merge_nodes >= num_old_nodes
    #            new_indices = torch.zeros(num_new_nodes, dtype=torch.long)
    #            new_indices[insert_indices] = 
    #            for feature_name in brick_list.brick_feature_names:
    #                # insert
    #                insert_features = brick_list[feature_name][insert_indices]
    #                insert_indices = merge_nodes[insert_indices]
    #                self[feature_name][insert_indices] = insert_features
    #                
    #                # append
    #                append_features = brick_list[feature_name][append_indices]
    #                self[feature_name] = torch.cat(
    #                        (self[feature_name], append_features))
    #            
    #            if new_new_edges is not None:
                    

def segmentation_to_brick_graph(scores, segmentation, feature_dict):
    # get dimensions (this works for either 3 or 4 dimensional tensors)
    b, h, w = scores.shape[0], scores.shape[-2], scores.shape[-1]
    scores = scores.view(b,h,w)

    # scatter_max the scores using the segmentation
    # raw_segment_score: the highest score for each segment
    # raw_source_index: the pixel location where raw_segment_score was found
    raw_segment_score, raw_source_index = scatter_max(
            scores.view(b,-1), segmentation.view(b,-1))

    # filter out the segments that have no contributing pixels
    # segment_score: filtered raw_segment_score
    # source_index: filtered raw_source_index
    valid_segments = torch.where(raw_segment_score > 0.)

    # jump through more hoops to use from_data_list because you can't break
    # the batch apart again later unless you do.
    # after looking at it... from_data_list screws up... all integers?!?
    graphs = []
    for i in range(b):
        item_entries = valid_segments[0] == i
        segment_ids = valid_segments[1][item_entries]
        num_segments = segment_ids.shape[0]
        batch_entries = [i] * num_segments

        item_segment_score = raw_segment_score[batch_entries, segment_ids]
        item_source_index = raw_source_index[batch_entries, segment_ids]

        positions = torch.stack(
                (item_source_index // w, item_source_index % w), dim=1)

        graph_data = {}
        for feature_name, feature_values in feature_dict.items():
            c = feature_values.shape[1]
            segment_features = feature_values.view(b, c, -1)[
                    batch_entries, :, item_source_index]
            graph_data[feature_name] = segment_features

        graphs.append(GraphData(
                edge_index=torch.zeros((2,0), dtype=torch.long),
                score=item_segment_score,
                segment_id=segment_ids,
                pos=positions,
                **graph_data))

    #batch = GraphBatch.from_data_list(graphs)
    #node_counts = [graph.num_nodes for graph in graphs]
    # this would not be necessary in the latest version of pytorch-geometric
    #batch.ptr = [sum(node_counts[:i]) for i in range(b+1)]
    return graphs

