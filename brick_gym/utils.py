import numpy

def matrix_to_edge_scores(image_index, node_label, edge_matrix):
    edge_scores = {}
    for i, label_i in enumerate(node_label):
        if label_i == 0:
            continue
        for j, label_j in enumerate(node_label[i+1:], start=i+1):
            if label_j == 0:
                continue
            if image_index is None:
                edge = (i+1, j+1, int(label_i), int(label_j))
            else:
                edge = (image_index, i+1, j+1, int(label_i), int(label_j))
            edge_scores[edge] = float(edge_matrix[i,j])
    
    return edge_scores

def sparse_graph_to_edge_scores(
        image_index,
        node_label,
        edges,
        scores,
        unidirectional,
        include_node_labels=True):
    
    if unidirectional:
        unidirectional_edges = edges[:,0] < edges[:,1]
        edges = edges[unidirectional_edges]
        scores = scores[unidirectional_edges]
    
    edge_scores = {}
    for (a, b), score in zip(edges, scores):
        score = float(score)
        
        key = []
        if image_index is not None:
            key.append(image_index)
        key.append(int(a))
        key.append(int(b))
        if include_node_labels:
            key.append(int(node_label[a]))
            key.append(int(node_label[b]))
        
        edge_scores[tuple(key)] = score
        
        '''
        if image_index is None:
            in include_node_labels:
                #edge_scores[a+1, b+1, node_label[a], node_label[b]] = score
                edge_scores[
                        int(a), int(b),
                        int(node_label[a]), int(node_label[b])] = score
            else:
                edge_scores[int(a), int(b)] = score
        else:
            #edge_scores[image_index,a+1,b+1,node_label[a],node_label[b]] = (
            #        score)
            edge_scores[image_index,
                    int(a), int(b),
                    int(node_label[a]), int(node_label[b])] = score
        '''
    
    return edge_scores

def sparse_graph_to_instance_map_scores(
        indices, instance_labels, scores):
    pass

def sparse_graph_to_instance_scores(
        image_index, indices, instance_labels, scores):
    instance_scores = {}
    for index, label, score in zip(indices, instance_labels, scores):
        label = int(label)
        if label == 0:
            continue
        if image_index is None:
            key = (index, label)
        else:
            key = (image_index, index, label)
        instance_scores[key] = score
    
    return instance_scores

def metadata_to_edge_scores(image_index, metadata):
    class_labels = metadata['class_labels']
    edges = metadata['edges']
    return {(a, b, class_labels[str(a)], class_labels[str(b)]) : 1.0
            for a, b in edges}

def metadata_to_graph(metadata, max_nodes=None):
    if max_nodes is None:
        max_nodes = max(int(key) for key in metadata['class_labels'])
    nodes = numpy.zeros(max_nodes, dtype=numpy.long)
    edges = numpy.zeros((max_nodes, max_nodes), dtype=numpy.float)
    for instance_id, class_label in metadata['class_labels'].items():
        instance_id = int(instance_id)-1
        nodes[instance_id] = class_label
    
    for (a,b) in metadata['edges']:
        a = a-1
        b = b-1
        edges[a,b] = 1.0
        edges[b,a] = 1.0
    
    return nodes, edges
