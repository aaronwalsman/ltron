import torch
from torchvision.transforms.functional import to_tensor

import tqdm

def train_greedy(
        env,
        model,
        optimizer,
        num_actions = 5,
        num_batches = 128,
        batch_size = 16):
    
    for batch in tqdm.tqdm(range(num_batches)):
        observations = []
        rewards = torch.zeros(batch_size, num_actions)
        for b in range(batch_size):
            observation = env.reset()
            state = env.get_state()
            for i, action in enumerate(range(num_actions)):
                _, reward, _, info = env.step(action)
                env.reset(state = state, render=False)
                rewards[b,i] = reward
            observations.append(to_tensor(observation))
        
        x = torch.stack(observations).cuda()
        rewards = rewards.cuda()
        
        y = torch.argmax(rewards, dim=-1)
        
        logits = model(x)
        loss = torch.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
