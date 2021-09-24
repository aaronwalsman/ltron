import random
import math
import json

import numpy

from gym.spaces import Dict, Discrete

from ltron.hierarchy import hierarchy_branch
from ltron.score import score_configurations
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.gym.spaces import ConfigurationSpace, InstanceMatchingSpace

class Reassembly(LtronGymComponent):
    def __init__(self,
        class_ids,
        color_ids,
        max_instances,
        max_edges,
        max_snaps_per_brick,
        workspace_scene_component,
        handspace_scene_component=None,
        dataset_component=None,
        reassembly_mode='clear',
        train=False,
        #wrong_pose_discount=0.1,
    ):
        #self.class_ids = class_ids
        #self.color_ids = color_ids
        self.max_instances = max_instances
        self.workspace_scene_component = workspace_scene_component
        self.handspace_scene_component = handspace_scene_component
        self.dataset_component = dataset_component
        self.reassembly_mode = reassembly_mode
        self.train = train
        #self.wrong_pose_discount = wrong_pose_discount
        
        #assert wrong_pose_discount > 0., 'Requires wrong pose discount > 0'
        
        #num_classes = max(self.class_ids.values())+1
        #num_colors = max(self.color_ids.values())+1
        
        self.action_space = Dict({'start':Discrete(2), 'end':Discrete(2)})
        observation_space = {'reassembling':Discrete(2)}
        if self.train:
            observation_space['target_configuration'] = ConfigurationSpace(
                #num_classes,
                #num_colors,
                class_ids,
                color_ids,
                max_instances,
                max_edges,
                max_snaps_per_brick,
            )
            observation_space['workspace_configuration'] = ConfigurationSpace(
                #num_classes,
                #num_colors,
                class_ids,
                color_ids,
                max_instances,
                max_edges,
                max_snaps_per_brick,
            )
            observation_space['handspace_configuration'] = ConfigurationSpace(
                #num_classes,
                #num_colors,
                class_ids,
                color_ids,
                1,
                0,
                max_snaps_per_brick,
            )
            observation_space['target_workspace_matching'] = (
                InstanceMatchingSpace(max_instances))
        
        self.observation_space = Dict(observation_space)
        self.reassembling=False
    
    def compute_observation(self):
        scene = self.workspace_scene_component.brick_scene
        handspace_scene = self.handspace_scene_component.brick_scene
        workspace_space = self.observation_space['workspace_configuration']
        #workspace_configuration = workspace_space.from_scene(
        #    scene, self.class_ids, self.color_ids)
        workspace_configuration = workspace_space.from_scene(scene)
        
        #current_bricks, current_neighbors = scene.get_brick_neighbors()
        
        #self.score, x_scores, y_scores, x_best, y_best, s = score_configurations(
        self.score, matching = score_configurations(
            #self.target_bricks,
            #self.target_neighbors,
            #current_bricks,
            #current_neighbors,
            self.target_configuration,
            workspace_configuration,
            #wrong_pose_discount=self.wrong_pose_discount,
        )
        
        #scene.export_ldraw('./tmp.mpd')
        
        self.observation = {'reassembling':self.reassembling}
        if self.train:
            self.observation['target_configuration'] = self.target_configuration
            self.observation['workspace_configuration'] = (
                workspace_configuration)
            handspace_space = self.observation_space['handspace_configuration']
            #handspace_configuration = handspace_space.from_scene(
            #    handspace_scene, self.class_ids, self.color_ids)
            handspace_configuration = handspace_space.from_scene(
                handspace_scene)
            self.observation['handspace_configuration'] = (
                handspace_configuration)
            #alignment = numpy.zeros((self.max_instances+1,), dtype=numpy.long)
            #alignment[1:len(x_best)+1] = x_best+1
            matching_array = numpy.zeros(
                (self.max_instances, 2), dtype=numpy.long)
            if len(matching):
                matching_array[:len(matching)] = list(matching)
            #alignment[:len(x_best),0] = x_best+1
            #alignment[:len(y_best),1] = y_best+1
            #score = numpy.zeros((self.max_instances, 2))
            #score[:len(x_scores),0] = x_scores
            #score[:len(y_scores),1] = y_scores
            #self.observation['target_workspace_alignment'] = {
            #    'alignment':alignment}
            self.observation['target_workspace_matching'] = matching_array
            
            #if self.dataset_component is not None:
            #    self.observation['target_ordering'] = (
            #        self.target_ordering)
    
    def reset(self):
        workspace_scene = self.workspace_scene_component.brick_scene
        #target_bricks, target_neighbors = workspace_scene.get_brick_neighbors()
        #self.target_bricks = [brick.clone() for brick in target_bricks]
        #self.target_neighbors = [
        #    [neighbor.clone() for neighbor in brick_neighbors]
        #    for brick_neighbors in target_neighbors
        #]
        if self.train:
            target_space = self.observation_space['target_configuration']
            #self.target_configuration = target_space.from_scene(
            #    workspace_scene, self.class_ids, self.color_ids)
            self.target_configuration = target_space.from_scene(workspace_scene)
            
            '''
            if self.dataset_component is not None:
                metadata_path = hierarchy_branch(
                    self.dataset_component.dataset_item, self.metadata_path)
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    orderings = metadata['reassembly_orderings']
                    ordering = random.choice(orderings)
                    self.target_ordering = numpy.zeros(
                        self.max_instances+1, dtype=numpy.long)
                    self.target_ordering[:len(ordering)] = ordering
            '''
        
        if self.handspace_scene_component is not None:
            handspace_scene = self.handspace_scene_component.brick_scene
            handspace_scene.clear_instances()
        
        self.reassembling=False
        
        self.compute_observation()
        
        return self.observation
    
    def step(self, action):
        
        if action['start'] and not self.reassembling:
            self.reassembling=True
            workspace_scene = self.workspace_scene_component.brick_scene
            workspace_scene.clear_instances()
            
            if self.handspace_scene_component is not None:
                handspace_scene = self.handspace_scene_component.brick_scene
                handspace_scene.clear_instances()
            
            if self.reassembly_mode == 'clear':
                pass
            
            elif self.reassembly_mode == 'square':
                square = math.ceil(len(self.target_bricks)**0.5)
                brick_order = list(range(len(self.target_bricks)))
                spacing=140
                for i, brick_id in enumerate(brick_order):
                    target_brick = self.target_bricks[brick_id]
                    x = i % square
                    z = i // square
                    transform = scene.upright.copy()
                    transform[0,3] = (x-square/2.) * spacing
                    transform[2,3] = (z-square/2.) * spacing
                    scene.add_instance(
                        target_brick.brick_type,
                        target_brick.color,
                        transform,
                    )
            
            else:
                raise NotImplementedError
        
        '''
        if self.reassembling:
            scene = self.workspace_scene_component.brick_scene
            #current_bricks, current_neighbors = scene.get_brick_neighbors()
            
            score, x_best, y_best = score_configurations(
                #self.target_bricks,
                #self.target_neighbors,
                #current_bricks,
                #current_neighbors,
                self.target_configuration,
                
            )
        else:
            score = 0.
        '''
        
        self.compute_observation()
        if self.reassembling:
            score = self.score
        else:
            score = 0.
        
        return self.observation, score, action['end'], {}
