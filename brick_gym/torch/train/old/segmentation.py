import torch
from torch.utils.data import DataLoader

import PIL.Image as Image

import tqdm

import brick_gym.config as config
import brick_gym.torch.models as models
from brick_gym.torch.datasets.segmentation import SegmentationDataset

def train_semantic_segmentation_epoch(
        model,
        loader,
        optimizer):
    
    model.train()
    train_iterate = tqdm.tqdm(loader)
    running_loss = None
    for images, instance_targets, class_targets in train_iterate:
        images = images.cuda()
        targets = targets.cuda()
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if running_loss is None:
            running_loss = float(loss)
        else:
            running_loss = running_loss * 0.9 + float(loss) * 0.1
        train_iterate.set_description('Loss: %.04f'%float(running_loss))

def test_semantic_segmentation_epoch(
        model,
        loader):
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        correct_foreground = 0
        total_foreground = 0
        test_iterate = tqdm.tqdm(loader)
        for i, (images, targets) in enumerate(test_iterate):
            images = images.cuda()
            targets = targets.cuda()
            logits = model(images)
            prediction = torch.argmax(logits, dim=1)
            correct += float(torch.sum(prediction == targets))
            total += float(targets.numel())
            
            foreground = targets != 0
            correct_foreground += float(torch.sum(
                    (prediction == targets) * foreground))
            total_foreground += float(torch.sum(foreground))
    
    print('Accuracy:            %f'%(correct/total))
    print('Foreground Accuracy: %f'%(correct_foreground/total_foreground))

def train_semantic_segmentation(
        num_epochs,
        model_name,
        dataset,
        train_subset=None,
        test_subset=None,
        batch_size=64,
        lr=3e-4,
        checkpoint_frequency=1,
        test_frequency=1):
    
    dataset_directory = config.datasets[dataset]
    print('Loading segmentation dataset from: %s'%dataset_directory)
    train_dataset = SegmentationDataset(
            dataset_directory, 'train', train_subset, include_class_labels=True)
    train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataset = SegmentationDataset(
            dataset_directory, 'test', test_subset, include_class_labels=True)
    test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    print('Building model: %s'%model_name)
    model = models.get_fcn_model(model_name, train_dataset.num_classes).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, num_epochs+1):
        print('Epoch: %i'%epoch)
        print('Train')
        train_semantic_segmentation_epoch(model, train_loader, optimizer)
        
        if checkpoint_frequency is not None and epoch%checkpoint_frequency == 0:
            model_checkpoint_path = './model_checkpoint_%04i.pt'%epoch
            print('Saving model to: %s'%model_checkpoint_path)
            torch.save(model.state_dict(), model_checkpoint_path)
            optimizer_checkpoint_path = './optimizer_checkpoint_%04i.pt'%epoch
            print('Saving optimzier to: %s'%optimizer_checkpoint_path)
            torch.save(optimizer.state_dict(), optimizer_checkpoint_path)
        
        if test_frequency is not None and epoch%test_frequency == 0:
            test_semantic_segmentation_epoch(model, test_loader)

def test_semantic_segmentation(
        model_name,
        dataset_directory,
        test_subset=None,
        batch_size=64):
    
    test_dataset = SegmentationDataset(
            dataset_directory, 'test', test_subset, include_class_labels=True)
    test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    model = models.get_fcn_model(model_name).cuda()
    
    test_semantic_segmentation_epoch(model, test_loader)
