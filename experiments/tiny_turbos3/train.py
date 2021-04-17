#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.graph as train_graph

#run = 'Feb24_11-25-43_mechagodzilla'
#epoch = 380
#run = 'Feb24_19-26-26_mechagodzilla' #None #'Feb23_00-57-59_mechagodzilla'
#epoch = 35
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
            num_epochs = 1000,
            mini_epochs_per_epoch = 1,
            
            # dataset settings
            dataset = 'tiny_turbos3',
            train_split = 'train',
            train_subset = None,
            num_processes = 8,
            randomize_viewpoint=True,
            random_floating_bricks=False,
            random_floating_pairs=False,
            random_bricks_rotation_mode='local_identity',
            
            # rollout settings
            train_steps_per_epoch = 1024,
            
            # train settings
            learning_rate = 1e-4,
            weight_decay = 1e-6,
            mini_epoch_sequences = 2048,
            mini_epoch_sequence_length = 1,
            batch_size = 32,
            #edge_loss_weight = 0.,
            #matching_loss_weight = 0.,
            multi_hide = True,
            max_instances_per_step=8,
            
            # model settings
            #step_model_backbone = 'smp_fpn_rnxt50',
            step_model_backbone = 'simple',
            edge_model_name = 'squared_difference',
            segment_id_matching = False,
            
            # test settings
            test_frequency = None,
            test_steps_per_epoch = 8, #512,
            
            # logging settings
            log_train=0,
            
            # checkpoint settings
            checkpoint_frequency=25)
