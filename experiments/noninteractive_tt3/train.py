#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.noninteractive_graph as train_graph

#run = 'Feb24_11-25-43_mechagodzilla'
#epoch = 380
#run = 'Feb24_19-26-26_mechagodzilla' #None #'Feb23_00-57-59_mechagodzilla'
#epoch = 35
run = None #'Mar07_23-27-29_mechagodzilla'
epoch = 0 # 20

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
            mini_epochs_per_epoch = 2,
            
            # dataset settings
            dataset = 'tiny_turbos3',
            train_split = 'train',
            train_subset = 4,
            num_processes = 4,
            randomize_viewpoint=True,
            random_floating_bricks=False,
            random_floating_pairs=False,
            random_bricks_rotation_mode='uniform',
            
            # rollout settings
            train_steps_per_epoch = 1024, 
            # train settings
            learning_rate = 1e-4,
            weight_decay = 1e-6,
            batch_size = 8,
            #edge_loss_weight = 10.,
            #matching_loss_weight = 0.,
            #--------------
            center_local_loss_weight = 0.05,
            center_cluster_loss_weight = 0.0, #1.0,
            center_separation_loss_weight = 1.0,
            center_separation_distance = 5,
            #--------------
            
            # model settings
            #step_model_backbone = 'smp_fpn_rnxt50',
            step_model_name = 'nth_try', #'center_voting',
            step_model_backbone = 'simple',
            edge_model_name = 'squared_difference',
            decoder_channels = 512,
            
            # logging settings
            log_train=8,
            
            # checkpoint settings
            checkpoint_frequency=5)
