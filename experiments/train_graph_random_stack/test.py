#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.test_graph as test_graph

#run = 'Jan28_22-18-53_mechagodzilla'
#epoch = 60
#run = 'Jan26_01-19-09_mechagodzilla'
#epoch = 55

run = 'Feb01_18-02-42_mechagodzilla'
epoch = 100

if __name__ == '__main__':
    test_graph.test_checkpoint(
            # load checkpoints
            step_checkpoint = './checkpoint/%s/step_model_%04i.pt'%(run, epoch),
            edge_checkpoint = './checkpoint/%s/edge_model_%04i.pt'%(run, epoch),
            
            # dataset settings
            dataset = 'random_stack',
            num_processes = 8,
            test_split = 'test',
            test_subset = 256,
            
            # model settings
            step_model_name='nth_try',
            segment_id_matching=True)
