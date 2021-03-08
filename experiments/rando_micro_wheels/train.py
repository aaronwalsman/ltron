#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.graph as train_graph

run = None #'Feb28_23-41-32_mechagodzilla'
epoch = 500

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
            dataset = 'rando_micro_wheels',
            train_split = 'all',
            train_subset = None,
            image_resolution = (384,384),
            num_processes = 16,
            randomize_viewpoint=True,
            
            #load_scenes=False,
            #random_floating_bricks=False,
            #random_floating_pairs=False,
            #random_bricks_rotation_mode='identity',
            #random_bricks_per_scene=(20,30),
            
            # rollout settings
            train_steps_per_epoch = 1024,
            
            # train settings
            learning_rate = 1e-2,
            weight_decay = 1e-6,
            mini_epoch_sequences = 2048,
            mini_epoch_sequence_length = 1,
            batch_size = 16,
            edge_loss_weight = 0.,
            matching_loss_weight = 0.,
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
            log_train=8,
            
            # checkpoint settings
            checkpoint_frequency=5)
