import random

import numpy

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class PartialDisassemblyComponent(LtronGymComponent):
    def __init__(self,
        disassembly_component,
        pos_snap_render_component,
        neg_snap_render_component,
        max_instances_per_scene,
        num_disassembly_steps=1,
    ):
        self.disassembly_component = disassembly_component
        self.pos_snap_render_component = pos_snap_render_component
        self.neg_snap_render_component = neg_snap_render_component
        self.num_disassembly_steps = num_disassembly_steps
        self.max_instances_per_scene = max_instances_per_scene
    
    def reset(self):
        for i in range(self.num_disassembly_steps):
            # get the rendered pos/neg snap images
            self.pos_snap_render_component.observe()
            self.neg_snap_render_component.observe()
            pos_render = self.pos_snap_render_component.observation
            neg_render = self.neg_snap_render_component.observation
            
            # turn the instance+snap tuples into single integers
            # and use numpy.unique to find all visible instance/snap pairs
            o = (self.max_instances_per_scene+1)
            pos_render = pos_render[...,0] + pos_render[...,1] * o
            neg_render = neg_render[...,0] + neg_render[...,1] * o
            visible_snap_instances = numpy.concatenate(
                (pos_render, neg_render), axis=0)
            visible_snap_instances = numpy.unique(visible_snap_instances)
            visible_instances = visible_snap_instances % o
            visible_snaps = visible_snap_instances // o
            visible_instance_snaps = [
                (i,s) for i,s in zip(visible_instances, visible_snaps)]
            
            # try to find a brick that is removable
            random.shuffle(visible_instance_snaps)
            for i, s in visible_instance_snaps:
                success, _, = self.disassembly_component.disassemble(i, s)
                if success:
                    break
            
            else:
                print('unable to remove anything?')
                import pdb
                pdb.set_trace()
