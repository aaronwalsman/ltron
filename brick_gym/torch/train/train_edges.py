#!/usr/bin/env python
import torch

import tqdm

import brick_gym.config as config
import brick_gym.random_stack.dataset as random_stack_dataset

class SimpleEdgeModel(torch.nn.Module):
    def __init__(self,
                num_x=32,
                num_y=8,
                num_z=32):
        super(SimpleEdgeModel, self).__init__()
        
        self.brick_embedding = torch.nn.Embedding(7, 128, padding_idx=0)
        self.x_embedding = torch.nn.Embedding(num_x, 128, padding_idx=0)
        self.y_embedding = torch.nn.Embedding(num_y, 128, padding_idx=0)
        self.z_embedding = torch.nn.Embedding(num_z, 128, padding_idx=0)
        self.o_embedding = torch.nn.Embedding(2, 128, padding_idx=0)
        
        self.linear_a = torch.nn.Linear(128*5, 512)
        self.linear_b = torch.nn.Linear(512, 512)
        
        self.combination_a = torch.nn.Linear(1024, 512)
        self.combination_b = torch.nn.Linear(512, 512)
        self.combination_c = torch.nn.Linear(512, 512)
        self.edge_out = torch.nn.Linear(512,2)
    
    def forward(self, bricks):
        batch_size, _, bricks_per_model = bricks.shape
        brick_embeddings = self.brick_embedding(bricks[:,0])
        x_embeddings = self.x_embedding(bricks[:,1])
        y_embeddings = self.y_embedding(bricks[:,2])
        z_embeddings = self.z_embedding(bricks[:,3])
        o_embeddings = self.x_embedding(bricks[:,4])
        brick_features = torch.cat((
                brick_embeddings,
                x_embeddings,
                y_embeddings,
                z_embeddings,
                o_embeddings), dim=2).view(-1, 128*5)
        brick_features = self.linear_a(brick_features)
        brick_features = torch.nn.functional.relu(brick_features)
        brick_features = self.linear_b(brick_features)
        brick_features = torch.nn.functional.relu(brick_features)
        
        brick_features_a = brick_features.reshape(
                batch_size, 1, bricks_per_model, 512).expand(
                batch_size, bricks_per_model, bricks_per_model, 512)
        brick_features_b = brick_features.reshape(
                batch_size, bricks_per_model, 1, 512).expand(
                batch_size, bricks_per_model, bricks_per_model, 512)
        
        edge_features = torch.cat(
                (brick_features_a, brick_features_b), dim=-1).view(-1, 1024)
        edge_features = self.combination_a(edge_features)
        edge_features = torch.nn.functional.relu(edge_features)
        edge_features = self.combination_b(edge_features)
        edge_features = torch.nn.functional.relu(edge_features)
        edge_features = self.combination_c(edge_features)
        edge_features = torch.nn.functional.relu(edge_features)
        return self.edge_out(edge_features).view(
                batch_size, bricks_per_model, bricks_per_model, 2)

train_dataset = random_stack_dataset.RandomStackEdgeDataset(
        config.paths['random_stack'], 'train')
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)

test_dataset = random_stack_dataset.RandomStackEdgeDataset(
        config.paths['random_stack'], 'test')
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=True)

model = SimpleEdgeModel().cuda()

optim = torch.optim.Adam(model.parameters(), lr=3e-4)

num_epochs = 50

for epoch in range(1, num_epochs+1):
    print('Epoch: %i'%epoch)
    print('Train')
    train_iterate = tqdm.tqdm(train_loader)
    for bricks, edges in train_iterate:
        logits = model(bricks).view(-1,2)
        loss = torch.nn.functional.cross_entropy(logits, edges.view(-1))
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        train_iterate.set_description('Loss: %.04f'%float(loss))
    
    print('Test')
    with torch.no_grad():
        correct = 0
        total = 0
        test_iterate = tqdm.tqdm(test_loader)
        for bricks, edges in test_iterate:
            logits = model(bricks)
            prediction = (logits[:,:,:,1] > logits[:,:,:,0]).long()
            correct += torch.sum(prediction == edges)
            total += edges.numel()
        
        print('Accuracy: %f'%(float(correct)/total))
