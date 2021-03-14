#!/usr/bin/env python
import argparse

from brick_gym.torch.train.segmentation import train_semantic_segmentation

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--model', type=str, default='smp_fpn_rnxt50')
    parser.add_argument(
            '--dataset', type=str, default='random_stack')
    parser.add_argument(
            '--batch-size', type=int, default=64)
    parser.add_argument(
            '--learning-rate', type=float, default=3e-4)
    parser.add_argument(
            '--num-epochs', type=int, default=25)
    parser.add_argument(
            '--train-subset', type=int, default=None)
    parser.add_argument(
            '--test-subset', type=int, default=None)
    parser.add_argument(
            '--checkpoint-frequency', type=int, default=1)
    parser.add_argument(
            '--test-frequency', type=int, default=1)
    args = parser.parse_args()

    train_semantic_segmentation(
            args.num_epochs,
            args.model,
            args.dataset,
            args.train_subset,
            args.test_subset,
            args.batch_size,
            args.learning_rate,
            args.checkpoint_frequency,
            args.test_frequency)
