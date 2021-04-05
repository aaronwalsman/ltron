import torch

from torch_scatter import scatter_max

from torch_geometric.data import Data as GraphData, Batch as GraphBatch

def segmentation_to_graph(scores, segmentation, feature_dict):
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
    
    '''
    segment_score = raw_segment_score[valid_segments]
    source_index = raw_source_index[valid_segments]
    
    # get the original 2D positions of the pixels contributing valid segments
    positions = torch.stack((source_index // w, source_index % w), dim=1)
    
    # build the graph by filtering each tensor in the feature dictionary
    graph_data = {}
    for feature_name, feature_values in feature_dict.items():
        c = feature_values.shape[1]
        segment_features = (
                feature_values.view(b, c, -1)[valid_segments[0],:,source_index])
        graph_data[feature_name] = segment_features
    
    # compute the ptr batch variable
    node_counts = [int(torch.sum(valid_segments[0] == i)) for i in range(b)]
    ptr = [sum(node_counts[:i]) for i in range(b+1)]
    
    # return the new graph batch
    return GraphBatch(
            batch=valid_segments[0],
            ptr=ptr,
            score=segment_score,
            segment_index=valid_segments[1],
            pos=positions,
            **graph_data)
    '''
