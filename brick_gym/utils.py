def matrix_to_edge_scores(image_index, node_label, edge_matrix):
    edge_scores = {}
    for i, label_i in enumerate(node_label):
        if label_i == 0:
            continue
        for j, label_j in enumerate(node_label[i+1:], start=i+1):
            if label_j == 0:
                continue
            if image_index is None:
                edge = (i, j, int(label_i), int(label_j))
            else:
                edge = (image_index, i, j, int(label_i), int(label_j))
            edge_scores[edge] = float(edge_matrix[i,j])
    
    return edge_scores
