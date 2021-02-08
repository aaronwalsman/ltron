#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import brick_gym.torch.train.test_graph as test_graph

#run = 'Jan24_01-28-20_mechagodzilla'
#epoch = 200
run = 'Feb04_15-46-25_mechagodzilla'
epoch = 45

if __name__ == '__main__':
    test_graph.test_checkpoint(
            # load checkpoints
            step_checkpoint = './checkpoint/%s/step_model_%04i.pt'%(run, epoch),
            edge_checkpoint = './checkpoint/%s/edge_model_%04i.pt'%(run, epoch),
            
            # dataset settings
            dataset = 'carbon_star',
            num_processes = 1,
            test_split = 'test',
            test_subset = None,
            
            # model settings
            step_model_name='nth_try',
            step_model_backbone='smp_fpn_r18',
            segment_id_matching=False)
