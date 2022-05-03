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

class MultiScreenPartialDisassemblyComponent(LtronGymComponent):
    def __init__(self,
        pick_and_place_component,
        pos_snap_render_component,
        neg_snap_render_component,
        pick_screen_names,
        place_screen_names,
        max_instances_per_scene,
        num_disassembly_steps=1,
    ):
        self.pick_and_place_component = pick_and_place_component
        self.pos_snap_render_component = pos_snap_render_component
        self.neg_snap_render_component = neg_snap_render_component
        self.pick_screen_names = pick_screen_names
        self.place_screen_names = place_screen_names
        self.num_disassembly_steps = num_disassembly_steps
        self.max_instances_per_scene = max_instances_per_scene
    
    def reset(self):
        for i in range(self.num_disassembly_steps):
            # get the rendered pos/neg snap images
            pos_render = self.pos_snap_render_component.observe()
            neg_render = self.neg_snap_render_component.observe()
            
            # turn the instance+snap tuples into single integers
            # and use numpy.unique to find all visible instance/snap pairs
            o = (self.max_instances_per_scene+1)
            pos_render = pos_render[...,0] + pos_render[...,1] * o
            neg_render = neg_render[...,0] + neg_render[...,1] * o
            visible_snap_instances = numpy.concatenate(
                (pos_render, neg_render), axis=0)
            #y, x = numpy.nonzero(visible_snap_instances)
            #nonzero_positions = list(zip(y, x))
            
            visible_snap_instances = numpy.unique(visible_snap_instances)
            visible_instances = visible_snap_instances % o
            visible_snaps = visible_snap_instances // o
            visible_instance_snaps = [
                (j,s) for j,s in zip(visible_instances, visible_snaps)
                if j != 0]
            
            # try to find a brick that is removable
            random.shuffle(visible_instance_snaps)
            for instance, snap in visible_instance_snaps:
                success = self.pick_and_place_component.pick_and_place(
                    self.pick_screen_names[i],
                    instance,
                    snap,
                    self.place_screen_names[i],
                    0,
                    0,
                )
                if success:
                    break
            #random.shuffle(nonzero_positions)
            #for pick_i, pick_s in pick_instances_and_snaps:
            #    success, _ = self.pick_and_place_component.pick_and_place(
            #        y,
            #        x,
            #        self.pick_screen_indices[i],
            #        0,
            #        0,
            #        self.place_screen_indices[i],
            #    )
            #    if success:
            #        break
            else:
                print('unable to remove anything?')
                import pdb
                pdb.set_trace()
