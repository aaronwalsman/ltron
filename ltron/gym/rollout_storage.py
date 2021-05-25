import random
import math
import copy

import numpy

import gym.spaces as spaces

def parallel_deepmap(fn, *a):
    if isinstance(a[0], dict):
        assert all(isinstance(aa, dict) for aa in a[1:])
        assert all(aa.keys() == a[0].keys() for aa in a[1:])
        return {
            key : parallel_deepmap(fn, *[aa[key] for aa in a])
            for key in a[0].keys()
        }
    
    elif isinstance(a[0], (tuple, list)):
        assert all(isinstance(aa, (tuple, list)) for aa in a[1:])
        assert all(len(aa) == len(a[0]) for aa in a[1:])
        return [
            parallel_deepmap(fn, *[aa[i] for aa in a])
            for i in range(len(a[0]))
        ]
    
    else:
        return fn(*a)

def concatenate_gym_data(*a, axis=0):
    def fn(*a):
        return numpy.concatenate(a, axis=axis)
    return parallel_deepmap(fn, *a)

def stack_gym_data(*a, axis=0):
    def fn(*a):
        return numpy.stack(a, axis=axis)
    return parallel_deepmap(fn, *a)

def extract_indices(a, ids):
    def fn(a):
        return a[ids]
    return parallel_deepmap(fn, a)

def pad_gym_data(a, pad, axis=0):
    def fn(*a):
        if a[0].shape[axis] < pad:
            pad_shape = list(a[0].shape)
            pad_shape[axis] = pad - a[0].shape[axis]
            z = numpy.zeros(pad_shape, dtype=a[0].dtype)
            return numpy.concatenate((a[0], z), axis=axis)
        
        else:
            return a[0]
    
    return parallel_deepmap(fn, a)

def get_gym_data_len(a, axis=0):
    class GymLenException(Exception):
        def __init__(self, gym_len):
            self.gym_len = gym_len
    def fn(*a):
        raise GymLenException(a[0].shape[axis])
    
    try:
        parallel_deepmap(fn, a)
    except GymLenException as e:
        return e.gym_len
    
    return 0

class RolloutStorage:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.gym_data = None
        
        self.seq_locations = {}
        self.batch_seq_ids = [None for _ in range(self.batch_size)]
        self.next_seq_index = 0
        self.total_steps = 0
    
    def __or__(self, other):
        assert self.batch_size == other.batch_size
        assert self.seq_locations == other.seq_locations
        assert self.total_steps == other.total_steps
        assert self.batch_seq_ids == other.batch_seq_ids
        assert self.next_seq_index == other.next_seq_index
        new_storage = RolloutStorage(self.batch_size)
        new_storage.seq_locations = copy.deepcopy(self.seq_locations)
        new_storage.batch_seq_ids = copy.deepcopy(self.batch_seq_ids)
        new_storage.next_seq_index = self.next_seq_index
        new_storage.total_steps = self.total_steps
        new_storage.gym_data = {}
        new_storage.gym_data.update(self.gym_data)
        new_storage.gym_data.update(other.gym_data)
        return new_storage
    
    def append_batch(self,
            valid=None,
            **kwargs):
        
        if valid is None:
            valid = [True for _ in range(self.batch_size)]
        
        if self.gym_data is None:
            self.gym_data = kwargs
        else:
            self.gym_data = concatenate_gym_data(self.gym_data, kwargs)
        
        for i, v in enumerate(valid):
            if v:
                step_index = self.total_steps + i
                if self.batch_seq_ids[i] not in self.seq_locations:
                    self.seq_locations[self.batch_seq_ids[i]] = []
                self.seq_locations[self.batch_seq_ids[i]].append(step_index)
            
        self.total_steps += self.batch_size
    
    def start_new_sequences(self, terminal, valid=None):
        if valid is None:
            valid = [True for _ in range(self.batch_size)]
        for i, (t, v) in enumerate(zip(terminal, valid)):
            if v and t:
                new_seq_index = self.next_seq_index
                self.next_seq_index += 1
                self.seq_locations[new_seq_index] = []
                self.batch_seq_ids[i] = new_seq_index
    
    def num_seqs(self):
        return len(self.seq_locations)
    
    def seq_len(self, seq):
        return len(self.seq_locations[seq])
    
    def get_storage_index(self, seq, step):
        return self.seq_locations[seq][step]
    
    def get_batch_from_storage_ids(self, storage_ids):
        gym_data = {}
        for key, value in self.gym_data.items():
            gym_data[key] = extract_indices(value, storage_ids)
        
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
    
    def batch_sequence_iterator(
        self,
        batch_size,
        max_seq_len=None,
        shuffle=False,
    ):
        return BatchSequenceIterator(
            self,
            batch_size,
            max_seq_len=max_seq_len,
            shuffle=shuffle,
        )
    
    def pad_stack_seqs(self, seq_ids, axis=1):
        if isinstance(seq_ids[0], int):
            gym_data = [self.get_seq(seq) for seq in seq_ids]
        else:
            gym_data = [
                self.get_seq(seq, start, stop)
                for seq, start, stop in seq_ids
            ]
        seq_lens = [get_gym_data_len(d) for d in gym_data]
        max_seq_len = max(seq_lens)
        gym_data = [pad_gym_data(d, max_seq_len) for d in gym_data]
        gym_data = stack_gym_data(*gym_data, axis=axis)
        seq_mask = numpy.ones(
            (len(seq_ids), max_seq_len),
            dtype=numpy.bool,
        )
        for i, seq_len in enumerate(seq_lens):
            seq_mask[i, :seq_len] = False
        return gym_data, seq_mask
    
    def get_current_seqs(self, stack_axis=1):
        return self.pad_stack_seqs(self.batch_seq_ids, axis=stack_axis)

class BatchSequenceIterator:
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
