import torch
import torchvision.models as torch_models

import tqdm

from brick_gym.torch.datasets.fixed_graph import FixedGraphDataset
import brick_gym.torch.models.resnet as bg_resnet
import brick_gym.torch.models.standard_models as standard_models

def train_fixed_graph(
        num_epochs,
        node_model_name,
        edge_model_name,
        dataset='connection2d',
        learning_rate=3e-4,
        batch_size=32):
    
    train_dataset = FixedGraphDataset(dataset, 'train')
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
    test_dataset = FixedGraphDataset(dataset, 'test')
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)
    
    node_model = standard_models.get_graphaction_model(
            node_model_name, classes=7).cuda()
    edge_model = standard_models.get_edge_model(edge_model_name).cuda()
    
    optimizer = torch.optim.Adam(
            list(node_model.parameters()) + list(edge_model.parameters()),
            lr=learning_rate)
    
    for epoch in range(1, num_epochs+1):
        print('Epoch %i'%epoch)
        train_iterate = tqdm.tqdm(train_loader)
        running_node_loss = 0.
        running_edge_loss = 0.
        for x, y_node, y_edge in train_iterate:
            x = x.cuda()
            y_node = y_node.cuda()
            y_edge = y_edge.cuda()
            f, node_logits, confidence_logits, action_logits = node_model(x)
            bs, num_instances, num_classes = node_logits.shape
            node_loss = torch.nn.functional.cross_entropy(
                    node_logits.view(-1, num_classes),
                    y_node.view(-1))
            running_node_loss = running_node_loss * 0.9 + float(node_loss) * 0.1
            
            edge_logits = edge_model(f)
            edge_loss = torch.nn.functional.binary_cross_entropy(
                    torch.sigmoid(edge_logits), y_edge.float())
            running_edge_loss = running_edge_loss * 0.9 + float(edge_loss) * 0.1
            
            loss = node_loss + edge_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_iterate.set_description(
                    'N:%.04f E:%.04f'%(running_node_loss, running_edge_loss))
