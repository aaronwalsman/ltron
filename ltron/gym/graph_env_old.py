import random
import os
import json

import numpy

import gym
import gym.spaces as spaces

import ltron.config as config
import ltron.evaluation as evaluation
from ltron.dataset.paths import (
        get_metadata, get_dataset_info, get_dataset_paths)
import ltron.dataset.ldraw_environment as ldraw_environment

class GraphEnv(gym.Env):
    def __init__(self,
            dataset,
            split,
            viewpoint_control,
            height=256,
            width=256,
            subset=None,
            rank=0,
            size=1,
            reward_mode='edge_ap',
            reset_mode='random'):
        
        self.model_paths = get_dataset_paths(dataset, split, subset, rank, size)
        self.episode_id = -1
        self.num_paths = len(self.model_paths)
        self.dataset_info = get_dataset_info(dataset)
        self.env = ldraw_environment.LDrawEnvironment(
                viewpoint_control, height, width)
        
        self.max_instances = self.dataset_info['max_instances_per_scene']
        self.num_classes = max(self.dataset_info['class_ids'].values())+1
        self.reward_mode = reward_mode
        self.reset_mode = reset_mode
        
        self.observation_space = spaces.Tuple((
                spaces.Box(
                    low=0,
                    high=255,
                    shape=(height, width, 3),
                    dtype=numpy.uint8),
                spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.max_instances, height, width, 3),
                    dtype=numpy.uint8)))
        
        self.action_space = spaces.Dict({
                'hide' : spaces.Discrete(self.max_instances+1),
                'node_class' : spaces.Box(
                    low=0,
                    high=self.num_classes,
                    shape=(self.max_instances,),
                    dtype=numpy.long),
                'edge_matrix' : spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_instances, self.max_instances),
                    dtype=numpy.long)})
        
        self.loaded_path = None
        self.metadata = None
        
        self.ground_truth_edges = set()
    
    def start_over(self, reset_mode=None):
        self.episode_id = -1
        if reset_mode is not None:
            self.reset_mode = reset_mode
    
    '''
    def compute_edge_ap(self, node_class, edge_matrix):
        # THIS IS BAD BECAUSE SCORES ARE ALWAYS 0/1?!?
        scores = []
        ground_truth = []
        predicted_edges = set()
        for i in range(edge_matrix.shape[0]):
            class_i = int(node_class[i])
            if class_i == 0:
                continue
            for j in range(i+1, edge_matrix.shape[1]):
                class_j = int(node_class[j])
                if class_j == 0:
                    continue
                if edge_matrix[i,j]:
                    edge = (class_i, class_j, i, j)
                    scores.append(1.0)
                    ground_truth.append(edge in self.ground_truth_edges)
                    predicted_edges.add(edge)
        false_negatives = len(self.ground_truth_edges - predicted_edges)
        _, _, ap = evaluation.ap(scores, ground_truth, false_negatives)
        
        return ap
    
    def compute_node_accuracy(self, node_class):
        #class_target = numpy.zeros(node_class.shape, dtype=numpy.long)
        #for instance, target in self.metadata['class_labels'].items():
        #    class_target[int(instance)-1] = target
        #return float(numpy.sum(node_class == class_target))/len(node_class)
        correct = 0
        total = 0
        #targets = []
        #predicted = []
        for instance, target in sorted(self.metadata['class_labels'].items()):
            total += 1
            correct += node_class[int(instance)-1] == target
            #targets.append(target)
            #predicted.append(node_class[int(instance)-1])
        #print('t/p', targets, predicted)
        return float(correct)/total
    '''
    
    def reward(self, node_class, edge_matrix):
        if self.reward_mode == 'edge_ap':
            return self.compute_edge_ap(node_class, edge_matrix)
        elif self.reward_mode == 'node_accuracy':
            return self.compute_node_accuracy(node_class)
    
    def observe(self):
        color = self.env.observe('color')
        self.recent_color = color
        instance_labels = self.env.observe('instance_labels')
        instance_segmentations = []
        for i in range(1, self.dataset_info['max_instances_per_scene']+1):
            mask = instance_labels == i
            mask = numpy.expand_dims(mask, -1)
            instance_segmentations.append(color * mask)
        
        return color, instance_segmentations
    
    def step(self, action):
        hide_index = action['hide']
        self.env.hide_brick(hide_index+1)
        observation = self.observe()
        if 'node_class' in action and 'edge_matrix' in action:
            reward = self.reward(action['node_class'], action['edge_matrix'])
        else:
            reward = 0
        
        self.step_index += 1
        if self.step_index == self.dataset_info['max_instances_per_scene']:
            terminal = True
        else:
            terminal = False
        return observation, reward, terminal, {}
    
    def get_node_and_edge_labels(self):
        return self.metadata['class_labels'], self.ground_truth_edges
    
    def get_hidden_nodes(self):
        return self.env.hidden_indices
    
    def reset(self):
        self.step_index = 0
        self.episode_id += 1
        if self.reset_mode == 'random':
            self.loaded_path = random.choice(self.model_paths)
        elif self.reset_mode == 'sequential':
            next_episode_id = self.episode_id % self.num_paths
            self.loaded_path = self.model_paths[next_episode_id]
        self.env.load_path(self.loaded_path)
        self.metadata = get_metadata(self.loaded_path)
        self.ground_truth_edges.clear()
        for a, b in self.metadata['edges']:
            class_a = self.metadata['class_labels'][str(a)]
            class_b = self.metadata['class_labels'][str(b)]
            self.ground_truth_edges.add((a,b,class_a,class_b))
        
        return self.observe()
    
    def render(self, mode='human', close=False):
        pass
