import torch
from torch.nn.functional import binary_cross_entropy

def dense_score_loss(
        dense_scores,
        correct,
        foreground,
        background_weight = 0.01,
        correct_weight = 0.1,
        incorrect_weight = 1.0):
    
    # get dimensions and reshape to batch_size x h x w
    # this removes any single channel
    b = dense_scores.shape[0]
    h, w = dense_scores.shape[-2:]
    dense_scores = dense_scores.view(b, h, w)
    correct = correct.view(b, h, w)
    
    # bce scores to correct
    score_loss = binary_cross_entropy(
            dense_scores,
            correct.float() * foreground,
            reduction = 'none')
    
    # reweight the different regions and sum
    #foreground_score_loss = score_loss * foreground
    summed_loss = torch.sum(score_loss * (
            foreground * correct * correct_weight +
            foreground * ~correct * incorrect_weight +
            ~foreground * background_weight))
    
    # compute the normalizer
    total_foreground_correct = torch.sum(foreground * correct)
    total_foreground_incorrect = torch.sum(foreground * ~correct)
    total_background = foreground.numel() - (
            total_foreground_correct + total_foreground_incorrect)
    normalizer = (
            total_foreground_correct * correct_weight +
            total_foreground_incorrect * incorrect_weight +
            total_background * background_weight)
    
    # normalize and return the loss
    normalized_loss = summed_loss / normalizer
    return normalized_loss

def cross_product_loss(
        scores,
        target,
        correct_weight = 1.0,
        incorrect_weight = 1.0):
    h, w = scores.shape[-2:]
    if h == 0 or w == 0:
        return 0.
    
    cross_loss = binary_cross_entropy(scores, target.float(), reduction='none')
    estimated_pos = scores > 0.5
    correct = estimated_pos == target
    
    #neg_weight = 1./(h*w)**0.5
    neg_weight = 1.0
    
    correct_pos = correct * target
    correct_neg = correct * ~target
    incorrect_pos = ~correct * target
    incorrect_neg = ~correct * ~target
    summed_loss = torch.sum(cross_loss * (
            correct_pos * correct_weight +
            correct_neg * correct_weight * neg_weight +
            incorrect_pos * incorrect_weight +
            incorrect_neg * incorrect_weight * neg_weight))
    
    # compute the normalizer
    total_correct_pos = torch.sum(correct_pos)
    total_correct_neg = torch.sum(correct_neg)
    total_incorrect_pos = torch.sum(incorrect_pos)
    total_incorrect_neg = torch.sum(incorrect_neg)
    normalizer = (
            total_correct_pos * correct_weight +
            total_correct_neg * correct_weight * neg_weight +
            total_incorrect_pos * incorrect_weight +
            total_incorrect_neg * incorrect_weight * neg_weight)
    
    # normalize and return
    normalized_loss = summed_loss / normalizer
    return normalized_loss
