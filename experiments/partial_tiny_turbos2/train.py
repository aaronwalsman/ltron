#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.graph as train_graph

run = None
epoch = 0

if run is not None:
    step_checkpoint = './checkpoint/%s/step_model_%04i.pt'%(run, epoch)
    edge_checkpoint = './checkpoint/%s/edge_model_%04i.pt'%(run, epoch)
    optimizer_checkpoint = './checkpoint/%s/optimizer_%04i.pt'%(run, epoch)
else:
    step_checkpoint = None
    edge_checkpoint = None
    optimizer_checkpoint = None

if __name__ == '__main__':
    train_graph.train_label_confidence(
            # load checkpoints
            step_checkpoint = step_checkpoint,
            edge_checkpoint = edge_checkpoint,
            optimizer_checkpoint = optimizer_checkpoint,
            
            # general settings
            num_epochs = 500,
            mini_epochs_per_epoch = 1,
            
            # dataset settings
            dataset = 'tiny_turbos2',
            train_subset = 4,
            num_processes = 4,
            randomize_viewpoint=True,
            
            # rollout settings
            train_steps_per_epoch = 1024,
            
            # train settings
            mini_epoch_sequences = 2048,
            mini_epoch_sequence_length = 1,
            batch_size = 6,
            #edge_loss_weight = 0.,
            #matching_loss_weight = 0.,
            
            # model settings
            step_model_backbone = 'smp_fpn_rnxt50',
            segment_id_matching = False,
            
            # test settings
            test_frequency = None,
            test_steps_per_epoch = 16, #512,
            
            # logging settings
            log_train=8,
            
            # checkpoint settings
            checkpoint_frequency=5)
