#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.graph as train_graph

run = None #'Feb01_22-37-07_gpu1'
epoch = 440

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
            mini_epoch_sequences = 2048*4,
            mini_epoch_sequence_length = 1,
            
            # dataset settings
            dataset = 'tiny_turbos2',
            num_processes = 8,
            randomize_viewpoint=True,
            
            # train settings
            train_steps_per_epoch = 4096,
            batch_size = 4,
            #edge_loss_weight = 0.,
            #matching_loss_weight = 0.,
            
            # model settings
            step_model_backbone = 'smp_fpn_r34',
            segment_id_matching = False,
            
            # test settings
            test_frequency = None,
            test_steps_per_epoch = 16, #512,
            
            # logging settings
            log_train=8,
            
            # checkpoint settings
            checkpoint_frequency=5)
