import torch

import PIL.Image as Image

import tqdm

import brick_gym.config as config

def train_epoch(
        model,
        loader,
        optimizer):
    
    model.train()
    train_iterate = tqdm.tqdm(train_loader)
    running_loss = None
    for images, targets in train_iterate:
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

def test_epoch(
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

def train_segmentation(
        num_epochs,
        dataset_args,
        lr=3e-4,
        checkpoint_frequency=1,
        test_frequency=1):
    
    train_loader = get_loader_somehow()
    test_loader = get_loader_somehow()
    
    model = get_model_somehow()
    
    optimizer = torch.optim.Adam(model.paramters(), lr=lr)
    
    for epoch in range(1, num_epochs+1):
        print('Epoch: %i'%epoch)
        print('Train')
        train_epoch(model, train_loader, optimizer)
        
        if checkpoint_frequency is not None and epoch%checkpoint_frequency == 0:
            checkpoint_path = './segmentation-checkpoint_%04i.pt'%epoch
            print('Saving checkpoint to: %s'%checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path)
        
        if test_frequency is not None and epoch%test_frequency == 0:
            test_epoch(model, test_loader)

def test_segmentation(
        ):
    
    test_loader = get_loader_somehow()
    
    model = get_model_somehow()
    
    test_epoch(model, test_loader)
