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
            dataset = 'micro_wheels',
            train_split = 'train',
            train_subset = None,
            num_processes = 4,
            randomize_viewpoint=True,
            random_floating_bricks=True,
            random_floating_pairs=False,
            random_bricks_rotation_mode='identity',
            load_scenes=False,
            random_bricks_per_scene=(20,30),
            
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
            log_train=8,
            
            # checkpoint settings
            checkpoint_frequency=5)