import torch

import torch_geometric.utils as tg_utils

from brick_gym.torch.brick_geometric import BrickGraph
import brick_gym.torch.models.mlp as mlp

class EdgeModel(torch.nn.Module):
    def __init__(self,
            in_channels,
            pre_compare_layers=3,
            post_compare_layers=3,
            compare_mode='add',
            split_heads=False,
            bias=True):
        
        super(EdgeModel, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = 2
        self.compare_mode = compare_mode
        self.split_heads = split_heads
        if bias:
            stdv = 1. / (in_channels**0.5)
            initial_bias = torch.zeros(1, in_channels).uniform_(-stdv,stdv)
            self.bias = torch.nn.Parameter(initial_bias)
        else:
            self.bias = None
        
        self.pre_compare = mlp.LinearStack(
                pre_compare_layers,
                in_channels,
                in_channels,
                in_channels)
        
        if self.split_heads:
            self.matching_head = mlp.LinearStack(
                    post_compare_layers,
                    in_channels,
                    in_channels,
                    1)
            self.edge_head = mlp.LinearStack(
                    post_compare_layers,
                    in_channels,
                    in_channels,
                    1)
        else:
            self.post_compare = mlp.LinearStack(
                    post_compare_layers,
                    in_channels,
                    in_channels,
                    2)
    
    def forward(self,
            step_lists,
            state_graphs,
            edge_threshold=0.05,
            max_edges=None,
            match_threshold=0.5,
            segment_id_matching=False):
        
        merged_graphs = []
        step_step_logits = []
        step_state_logits = []
        for step_list, state_graph in zip(step_lists, state_graphs):
            # -----------------------
            # compute step-step edges
            
            # pre-compare step_list
            step_x = step_list.x
            step_x = self.pre_compare(step_x)
            step_num, channels = step_x.shape
            
            # compare step_x with step_x
            step_one_x = step_x.view(step_num, 1, channels)
            one_step_x = step_x.view(1, step_num, channels)
            if self.compare_mode == 'add':
                step_step_x = step_one_x + one_step_x
            elif self.compare_mode == 'subtract':
                step_step_x = step_one_x - one_step_x
            elif self.compare_mode == 'squared_difference':
                step_step_x = (step_one_x - one_step_x)**2
            else:
                raise ValueError('Unknown compare mode: %s'%self.compare_mode)
        
            # post-compare step_step_x
            if self.split_heads:
                step_match_step_logits = self.matching_head(
                        step_step_x.view(step_num**2, channels))
                step_match_step_logits = step_match_step_logits.view(
                        step_num, step_num)
                
                step_edge_step_logits = self.edge_head(
                        step_step_x.view(step_num**2, channels))
                step_edge_step_logits = step_edge_step_logits.view(
                        step_num, step_num)
                
                step_step_x = torch.stack(
                        (step_match_step_logits, step_edge_step_logits), dim=-1)
            
            else:
                step_step_x = self.post_compare(
                        step_step_x.view(step_num**2, channels))
                step_step_x = step_step_x.view(step_num, step_num, 2)
                step_match_step_logits = step_step_x[..., 0]
                step_edge_step_logits = step_step_x[..., 1]
                #step_step_score = torch.sigmoid(step_step_x)
                #step_edge_step = step_step_score[...,1]
                #step_match_step = step_step_score[...,1] # unused
            
            # sparsify edge list and edge scores
            step_edge_step_scores = torch.sigmoid(step_edge_step_logits)
            sparse_step_edge_step_entries = tg_utils.dense_to_sparse(
                    step_edge_step_scores > edge_threshold)[0]
            
            sparse_step_edge_step_scores = step_edge_step_scores[
                    sparse_step_edge_step_entries[0],
                    sparse_step_edge_step_entries[1]].view(-1,1)
            
            # ----------------
            # build step_graph
            step_graph = BrickGraph(
                    step_list,
                    sparse_step_edge_step_entries,
                    sparse_step_edge_step_scores)
            
            # ------------------------
            # compute step-state edges
            
            # pre-compare state_graph
            state_x = state_graph.x
            state_x = self.pre_compare(state_x)
            state_num, state_channels = state_x.shape
            
            # compare step_x with state_x
            one_state_x = state_x.view(1, state_num, channels)
            if self.compare_mode == 'add':
                step_state_x = step_one_x + one_state_x
            elif self.compare_mode == 'subtract':
                step_state_x = step_one_x - one_state_x
            elif self.compare_mode == 'squared_difference':
                step_state_x = (step_one_x - one_state_x)**2
            else:
                raise ValueError('Unknown compare mode: %s'%self.compare_mode)
            
            # post-compare step_state_x
            if self.split_heads:
                step_match_state_logits = self.matching_head(
                        step_state_x.view(step_num*state_num, channels))
                step_match_state_logits = step_match_state_logits.view(
                        step_num, state_num)
                
                step_edge_state_logits = self.edge_head(
                        step_state_x.view(step_num*state_num, channels))
                step_edge_state_logits = step_edge_state_logits.view(
                        step_num, state_num)
                
                step_state_x = torch.stack(
                        (step_match_state_logits, step_edge_state_logits),
                        dim=-1)
            else:
                step_state_x = self.post_compare(
                        step_state_x.view(step_num*state_num, channels))
                step_state_x = step_state_x.view(step_num, state_num, 2)
                
                #step_state_score = torch.sigmoid(step_state_x)
                #step_match_state = step_state_score[...,0]
                #step_edge_state = step_state_score[...,1]
                step_match_state_logits = step_state_x[...,0]
                step_edge_state_logits = step_state_x[...,1]
            
            # sparsify edge list and edge scores
            step_edge_state_scores = torch.sigmoid(step_edge_state_logits)
            sparse_step_edge_state_entries = tg_utils.dense_to_sparse(
                    step_edge_state_scores > edge_threshold)[0]
            
            sparse_step_edge_state_scores = step_edge_state_scores[
                    sparse_step_edge_state_entries[0],
                    sparse_step_edge_state_entries[1]].view(-1,1)
            
            # ----------------------
            # compute matching nodes
            
            # matching must be (zero-or-one)-to-(zero-or-one)
            # some kind of Hungarian thing is probably the right thing to
            # do here but instead for now I'm just zero-ing out the non-max
            # of each row then the non-max of each column.
            if step_graph.num_nodes and state_graph.num_nodes:
                if segment_id_matching:
                    step_segment_id = step_list.segment_id.view(-1,1)
                    state_segment_id = state_graph.segment_id.view(1,-1)
                    segment_id_match = step_segment_id == state_segment_id
                    sparse_step_match_state_entries = torch.nonzero(
                            segment_id_match, as_tuple=False).t()
                    
                else:
                    step_match_state_scores = torch.sigmoid(
                            step_match_state_logits)
                     
                    step_argmax = torch.argmax(step_match_state_scores, dim=0)
                    mask = torch.zeros_like(step_match_state_scores)
                    mask[step_argmax, range(state_graph.num_nodes)] = 1.
                    step_match_state_scores = step_match_state_scores * mask
                    
                    state_argmax = torch.argmax(step_match_state_scores, dim=1)
                    mask = torch.zeros_like(step_match_state_scores)
                    mask[range(step_graph.num_nodes), state_argmax] = 1.
                    step_match_state_scores = step_match_state_scores * mask
                    
                    sparse_step_match_state_entries = torch.nonzero(
                            step_match_state_scores > match_threshold,
                            as_tuple=False).t()
            else:
                sparse_step_match_state_entries = None
            
            # --------------------
            # compute merged graph
            merged_graph = state_graph.merge(
                    step_graph,
                    sparse_step_match_state_entries,
                    sparse_step_edge_state_entries,
                    sparse_step_edge_state_scores)
            
            num_edges = merged_graph.edge_attr.shape[0]
            if max_edges is not None and num_edges > max_edges:
                #print('clipping edges')
                topk_v, topk_i = torch.topk(
                        merged_graph.edge_attr[:,0], max_edges)
                merged_graph.edge_index = (
                        merged_graph.edge_index[:,topk_i])
                merged_graph.edge_attr = (
                        merged_graph.edge_attr[topk_i])
            
            merged_graphs.append(merged_graph)
            step_step_logits.append(step_step_x)
            step_state_logits.append(step_state_x)
        
        return merged_graphs, step_step_logits, step_state_logits
        
        '''
        if graph_b is not None:
            x_b = graph_b.x
            x_b = self.pre_compare(x_b)
            n_b, c_b = x_b.shape
        else:
            x_b, n_b, c_b = x_a, n_a, c_a
        
        # compare
        assert c_a == c_b
        x_a = x_a.view(n_a, 1, c_a)
        x_b = x_b.view(1, n_b, c_b)
        if self.compare_mode == 'add':
            x = x_a + x_b
        elif self.compare_mode == 'subtract':
            x = x_a - x_b
        if self.bias is not None:
            x = x + self.bias
        
        # post-compare
        x = self.post_compare(x.view(n_a * n_b, c_a)).view(n_a, n_b, -1)
        return x
        '''

'''
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels)
    
    def forward(x, edge_index):
        x = self.linear(x)
        return self.propagate(edge_index, x)

class EdgeModel(torch.nn.Module):
    def __init__(self, in_channels, graph_model):
        super(EdgeModel, self).__init__()
        self.in_channels = in_channels
        self.graph_model = graph_model
        stdv = 1. / (self.in_channels**0.5)
        initial_edge_data = torch.zeros(
                1, self.in_channels).uniform_(-stdv,stdv)
        self.edge_bias = torch.nn.Parameter(initial_edge_data)
    
    def forward(self, graph_a, graph_b):
        # TODO: make it batch
        num_a = graph_a.num_nodes
        num_b = graph_b.num_nodes
        num_edges = num_a * num_b
        edge_x = self.edge_bias.expand(num_edges, self.in_channels)
        x = torch.cat((graph_a.x, graph_b.x, edge_x), dim=0)
        edge_index = torch.zeros(
                (2, graph_a.num_nodes, graph_b.num_nodes), dtype=torch.long)
        edge_index[0,:] = torch.arange(graph_a.num_nodes).unsqueeze(-1)
        edge_index[0,:] = torch.arange(graph_b.num_nodes).unsqueeze(0) + num_a
        edge_index = edge_index.view(2,-1)
        edge_index[1,:] = (
                torch.arange(graph_a.num_nodes * graph_b.num_nodes) +
                num_a + num_b)
        edge_index = edge_index.to(x.device)
        edge_features = self.graph_model(x, edge_index=edge_index)[num_a+num_b:]
        
        return edge_features.view(num_a, num_b, -1)
'''
