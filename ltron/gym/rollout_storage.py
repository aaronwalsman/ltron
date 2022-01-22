import random
import math
import copy
import os

import numpy

import gym.spaces as spaces

from ltron.hierarchy import (
    map_hierarchies,
    index_hierarchy,
    len_hierarchy,
    concatenate_numpy_hierarchies,
    stack_numpy_hierarchies,
    pad_numpy_hierarchy,
    hierarchy_branch,
    increase_capacity,
    set_index_hierarchy,
)

class RolloutStorage:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.gym_data = None
        
        self.seq_locations = {}
        self.finished_seqs = set()
        self.batch_seq_ids = [None for _ in range(self.batch_size)]
        self.next_seq_index = 0
        self.total_steps = 0
        self.batch_index = 0
    
    def save(self, path, finished_only=False, seq_ids=None):
        path = os.path.expanduser(path)
        
        if seq_ids is None:
            seq_ids = self.seq_locations.keys()
        
        if finished_only:
            seq_ids = [s for s in seq_ids if s in self.finished_seqs]
        
        for i, seq_id in enumerate(seq_ids):
            seq_path = os.path.join(path, 'seq_%06i.npz'%i)
            seq = self.get_seq(seq_id)
            numpy.savez_compressed(seq_path, seq=seq)
    
    def __or__(self, other):
        assert self.batch_size == other.batch_size
        assert self.seq_locations == other.seq_locations
        assert self.total_steps == other.total_steps
        assert self.batch_seq_ids == other.batch_seq_ids
        assert self.next_seq_index == other.next_seq_index
        assert self.finished_seqs == other.finished_seqs
        new_storage = RolloutStorage(self.batch_size)
        new_storage.seq_locations = copy.deepcopy(self.seq_locations)
        new_storage.batch_seq_ids = copy.deepcopy(self.batch_seq_ids)
        new_storage.next_seq_index = self.next_seq_index
        new_storage.total_steps = self.total_steps
        # THIS IS WRONG, SHOULD JUST USE SOME HIERARCHY CONCAT THING
        new_storage.gym_data = {}
        new_storage.gym_data.update(self.gym_data)
        new_storage.gym_data.update(other.gym_data)
        new_storage.finished_seqs = self.finished_seqs
        return new_storage
    
    def __getitem__(self, branch_keys):
        sub_storage = ReadOnlyRolloutStorage(self.batch_size)
        sub_storage.gym_data = hierarchy_branch(self.gym_data, branch_keys)
        sub_storage.seq_locations = self.seq_locations
        sub_storage.batch_seq_ids = self.batch_seq_ids
        sub_storage.next_seq_index = self.next_seq_index
        sub_storage.total_steps = self.total_steps
        
        return sub_storage
    
    def append_batch(self, valid=None, **kwargs):
        
        if valid is None:
            valid = [True for _ in range(self.batch_size)]
        
        if self.gym_data is None:
            self.gym_data = kwargs
            self.batch_index += self.batch_size
        else:
            #self.gym_data = concatenate_numpy_hierarchies(
            #    self.gym_data, kwargs)
            if self.batch_index >= len_hierarchy(self.gym_data):
                self.gym_data = increase_capacity(self.gym_data, factor=2)
            set_index_hierarchy(
                self.gym_data, kwargs,
                range(self.batch_index, self.batch_index+self.batch_size),
            )
            self.batch_index += self.batch_size
        
        for i, v in enumerate(valid):
            step_index = self.total_steps + i
            if v:
                if self.batch_seq_ids[i] not in self.seq_locations:
                    self.seq_locations[self.batch_seq_ids[i]] = []
                self.seq_locations[self.batch_seq_ids[i]].append(step_index)
            
        self.total_steps += self.batch_size
    
    def start_new_seqs(self, terminal, valid=None):
        if valid is None:
            valid = [True for _ in range(self.batch_size)]
        for i, (t, v) in enumerate(zip(terminal, valid)):
            if v and t:
                new_seq_index = self.next_seq_index
                self.next_seq_index += 1
                self.seq_locations[new_seq_index] = []
                finished_seq = self.batch_seq_ids[i]
                if finished_seq is not None:
                    self.finished_seqs.add(finished_seq)
                self.batch_seq_ids[i] = new_seq_index
    
    def num_seqs(self):
        return len(self.seq_locations)
    
    def num_finished_seqs(self):
        return len(self.finished_seqs)
    
    def seq_len(self, seq):
        return len(self.seq_locations[seq])
    
    def get_current_seq_lens(self):
        return [self.seq_len(seq) for seq in self.batch_seq_ids]
    
    def get_storage_index(self, seq, step):
        return self.seq_locations[seq][step]
    
    def get_batch_from_storage_ids(self, storage_ids):
        # WHY IS THIS IF/ELSE NECESSARY?  Can't index_hierarchy handle this?
        if isinstance(self.gym_data, dict):
            gym_data = {}
            for key, value in self.gym_data.items():
                gym_data[key] = index_hierarchy(value, storage_ids)
        else:
            gym_data = index_hierarchy(self.gym_data, storage_ids)
        
        return gym_data
    
    def get_batch(self, seq_step_ids):
        storage_ids = [
            self.get_storage_location(seq, step)
            for seq, step in seq_step_ids
        ]
        return self.get_batch_from_storage_ids(storage_ids)
    
    def get_seq(self, seq, start=None, stop=None):
        storage_ids = self.seq_locations[seq][start:stop]
        return self.get_batch_from_storage_ids(storage_ids)
    
    def batch_seq_iterator(
        self,
        batch_size,
        max_seq_len=None,
        shuffle=False,
    ):
        return BatchSeqIterator(
            self,
            batch_size,
            max_seq_len=max_seq_len,
            shuffle=shuffle,
        )
    
    def pad_stack_seqs(self, seq_ids, axis=1, start=None, stop=None):
        if isinstance(seq_ids[0], int):
            gym_data = [
                self.get_seq(seq, start=start, stop=stop) for seq in seq_ids]
        else:
            gym_data = [
                self.get_seq(seq, start, stop)
                for seq, start, stop in seq_ids
            ]
        seq_lens = numpy.array(
            [len_hierarchy(d) for d in gym_data], dtype=numpy.long)
        max_seq_len = max(seq_lens)
        gym_data = [pad_numpy_hierarchy(d, max_seq_len) for d in gym_data]
        gym_data = stack_numpy_hierarchies(*gym_data, axis=axis)
        return gym_data, seq_lens
    
    def get_current_seqs(self, stack_axis=1, start=None, stop=None):
        return self.pad_stack_seqs(
            self.batch_seq_ids, axis=stack_axis, start=start, stop=stop)
    
    def chop_sequences(self, max_seq_len=None):
        seq_ids = list(range(self.num_seqs()))
        seq_id_start_stops = []
        for seq_id in seq_ids:
            seq_len = self.seq_len(seq_id)
            if max_seq_len is None or seq_len < max_seq_len:
                seq_id_start_stops.append((seq_id, None, None))
            else:
                start = 0
                while start < seq_len:
                    stop = start + max_seq_len
                    seq_id_start_stops.append((seq_id, start, stop))
                    start = stop
        
        return seq_id_start_stops

class ReadOnlyRolloutStorage(RolloutStorage):
    def append_batch(self, *args, **kwargs):
        raise Exception('Attempted to append batch to read only storage')
    
    def start_new_seqs(self, *args, **kwargs):
        raise Exception('Attempted to start new seqs on read only storage')

class BatchSeqIterator:
    def __init__(
        self,
        rollout_storage,
        batch_size,
        max_seq_len=None,
        shuffle=False
    ):
        self.rollout_storage = rollout_storage
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        
        '''
        seq_ids = list(range(rollout_storage.num_seqs()))
        self.seq_id_start_stops = []
        for seq_id in seq_ids:
            seq_len = rollout_storage.seq_len(seq_id)
            if max_seq_len is None or seq_len < max_seq_len:
                self.seq_id_start_stops.append((seq_id, None, None))
            else:
                start = 0
                while start < seq_len:
                    stop = start + max_seq_len
                    self.seq_id_start_stops.append((seq_id, start, stop))
                    start = stop
        '''
        self.seq_id_start_stops = self.rollout_storage.chop_sequences(
            max_seq_len)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.seq_id_start_stops)
        self.batch_start = 0
        return self
    
    def __next__(self):
        if self.batch_start >= len(self.seq_id_start_stops):
            raise StopIteration
        
        batch_end = self.batch_start + self.batch_size
        batch_seq_ids = self.seq_id_start_stops[self.batch_start:batch_end]
        
        gym_data, seq_mask = self.rollout_storage.pad_stack_seqs(batch_seq_ids)
        
        self.batch_start += self.batch_size
        
        return gym_data, seq_mask
    
    def __len__(self):
        return math.ceil(len(self.seq_id_start_stops) / self.batch_size)
