from torch.distributions import Categorical, Bernoulli

from torch_geometric.data import Batch as GraphBatch

def remap_node_indices(batch_graph, indices, remaps):
    remap_split = split_node_value(batch_graph, remaps)
    return [remap[index] for remap, index in zip(remap_split, indices)]

def index_node_value(batch_graph, value, index):
    if isinstance(value, str):
        value = batch_graph[value]
    return value[batch_graph.ptr[index]:batch_graph.ptr[index+1]]

def split_node_value(batch_graph, value):
    return [index_node_value(batch_graph, value, index)
            for index in range(batch_graph.num_graphs)]

def batch_graph_distributions(
        Distribution, batch_graph, probs=None, logits=None, validate_args=None):
    distributions = []
    if isinstance(batch_graph, GraphBatch):
        for b in range(batch_graph.num_graphs):
            p = None
            if probs is not None:
                p = index_node_value(batch_graph, probs, b)
            l = None
            if logits is not None:
                l = index_node_value(batch_graph, logits, b)
            distributions.append(Distribution(
                    probs=p, logits=l, validate_args=validate_args))
    else:
        for graph in batch_graph:
            p = None
            if probs is not None:
                p = index
    return distributions

def batch_graph_categoricals(
        batch_graph, probs=None, logits=None, validate_args=None):
    return batch_graph_distributions(
        Categorical, batch_graph, probs, logits, validate_args)

def batch_graph_bernoullis(
        batch_graph, probs=None, logits=None, validate_args=None):
    return batch_graph_distributions(
        Bernoulli, batch_graph, probs, logits, validate_args)

def batch_graph_to_instance_edge_dicts(batch_graph, node_labels):
    for graph in batch_graph.to_data_list():
        print('---')
        print(graph.segment_index)
        print(graph.edge_index)
        print(graph.segment_index[graph.edge_index])

def insert_nodes(
        grapb_a,
        graph_b,
        comparison_key,
        score_key,
        comparison_fn = lambda a, b : a == b,
        overwrite_mode = 'max'):
    
    a_features = graph_a[comparison_key]
    b_features = graph_b[comparison_key]
    
    # keep track of:
    #   new node locations
