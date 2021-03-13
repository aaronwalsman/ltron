#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.test_graph_b as test_graph_b

#run = 'Jan24_01-28-20_mechagodzilla'
#epoch = 200
#run = 'Feb10_11-50-54_gpu3'
#run = 'Feb23_00-57-59_mechagodzilla' #'Feb17_22-57-19_gpu3'
#run = 'Mar05_21-48-25_mechagodzilla'
run = 'Mar08_00-45-28_mechagodzilla'
epoch = 250

if __name__ == '__main__':
    test_graph_b.test_checkpoint(
            # load checkpoints
            step_checkpoint = './checkpoint/%s/step_model_%04i.pt'%(run, epoch),
            edge_checkpoint = './checkpoint/%s/edge_model_%04i.pt'%(run, epoch),
            
            # dataset settings
            dataset = 'tiny_turbos2',
            num_processes = 4,
            test_split = 'test',
            test_subset = 4,
            
            # model settings
            step_model_name='center_voting', #'nth_try',
            decoder_channels=512,
            step_model_backbone='simple',
            segment_id_matching=False,
            
            # output settings
            dump_debug=True)
