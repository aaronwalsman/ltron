#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.graph as train_graph

if __name__ == '__main__':
    train_graph.train_label_confidence(
            # load checkpoints
            #step_checkpoint = './checkpoint/Jan23_14-30-07_mechagodzilla/'
            #        'step_model_0010.pt',
            #edge_checkpoint = './checkpoint/Jan23_14-30-07_mechagodzilla/'
            #        'edge_model_0010.pt',
            #optimizer_checkpoint = './checkpoint/Jan23_14-30-07_mechagodzilla/'
            #        'optimizer_0010.pt',
            
            # general settings
            num_epochs = 500,
            mini_epochs_per_epoch = 1,
            mini_epoch_sequences = 512,
            mini_epoch_sequence_length = 1,
            
            # dataset settings
            dataset = 'tiny_turbos2',
            num_processes = 8,
            randomize_viewpoint=True,
            
            # train settings
            train_steps_per_epoch = 1024,
            batch_size = 6,
            edge_loss_weight = 0.,
            matching_loss_weight = 0.,
            
            # model settings
            step_model_backbone = 'smp_fpn_r18',
            segment_id_matching = True,
            
            # test settings
            test_frequency = None,
            test_steps_per_epoch = 16, #512,
            
            # logging settings
            log_train=8,
            
            # checkpoint settings
            checkpoint_frequency=5)
