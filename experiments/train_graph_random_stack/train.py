#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.graph as train_graph

if __name__ == '__main__':
    train_graph.train_label_confidence(
            train_steps_per_epoch = 1024,
            #train_subset = 1,
            test_steps_per_epoch = 16, #512,
            batch_size = 2,
            num_processes = 8,
            num_epochs = 500,
            mini_epoch_sequences = 2048,
            mini_epoch_sequence_length = 2,
            mini_epochs_per_epoch = 1,
            dataset = 'random_stack',
            randomize_viewpoint=False,
            log_train=8,
            checkpoint_frequency=5)
