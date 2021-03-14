#!/usr/bin/env python
import argparse

import brick_gym.torch.train.graph_b as graph_b

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='random_stack')
    parser.add_argument('--train-split', type=str, default='train_mpd')
    parser.add_argument('--test-split', type=str, default='test_mpd')
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--test-checkpoint', type=str, default=None)
    parser.add_argument('--input-mode', type=str, default='images')
    parser.add_argument('--num-processes', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=3e-2)

    args = parser.parse_args()
    num_processes = args.num_processes
    if num_processes is None:
        if args.input_mode == 'images':
            num_processes = 4
        else:
            num_processes = 16

    if args.test_checkpoint is None:
        graph_b.train_hide_reinforce(
                3000,
                args.dataset,
                train_split=args.train_split,
                checkpoint_frequency=25,
                input_mode=args.input_mode,
                learning_rate = args.learning_rate,
                num_processes=num_processes)
    else:
        graph_b.test_hide_reinforce(
                test_checkpoint = args.test_checkpoint,
                dataset = args.dataset,
                test_split = args.test_split,
                subset = args.subset,
                input_mode=args.input_mode,
                num_processes=num_processes)
