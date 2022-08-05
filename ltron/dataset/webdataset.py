from io import BytesIO

import numpy

from webdataset import WebDataset

import ltron.settings as settings
from ltron.hierarchy import auto_pad_stack_numpy_hierarchies
from ltron.dataset.info import get_split_shards

def standard_transforms(
    dataset,
    subset=None,
    rank=0,
    size=1,
    shuffle=False,
    shuffle_buffer=1000,
    repeat=False,
):
    if subset is not None:
        dataset = dataset.slice(subset)
    if rank != 0 and size != 1:
        dataset = dataset.slice(rank, None, size)
    dataset = dataset.shuffle(shuffle * shuffle_buffer)
    if repeat:
        dataset = dataset.repeat()
    
    return dataset

def get_mpd_webdataset(
    dataset,
    split,
    **kwargs
):
    shards = get_split_shards(dataset, split)
    shards = [settings.shards[shard] for shard in shards]
    return get_mpd_webdataset_from_shards(shards, **kwargs)

def get_mpd_webdataset_from_shards(
    shards,
    **kwargs,
):
    dataset = WebDataset(shards).rename(mpd='mpd;ldr;l3b')
    dataset = standard_transforms(dataset, **kwargs)
    
    return dataset

def get_episode_webdataset(
    dataset,
    split,
    batch_size=None,
    **kwargs,
):
    shards = get_split_shards(dataset, split)
    shards = [settings.shards[shard] for shard in shards]
    return get_episode_dataset_from_shards(**kwargs)

def get_episode_webdataset_from_shards(shards, batch_size=None, **kwargs):
    
    def npz_extractor(item):
        data = BytesIO(item['npz'])
        data = numpy.load(data, allow_pickle=True)['seq'].item()
        return data
    
    def tuplefy(item):
        return (item,)
    
    def collate(item):
        item = [{k:v for k,v in i.items() if k != '__key__'} for i in item]
        return auto_pad_stack_numpy_hierarchies(*item, pad_axis=0, stack_axis=1)
    
    dataset = WebDataset(shards).map(npz_extractor)
    dataset = standard_transforms(dataset, **kwargs)
    if batch_size is not None:
        #dataset = dataset.map(tuplefy).batched(batch_size).map(collate)
        #dataset = dataset.batched(batch_size).map(collate)
        dataset = dataset.batched(batch_size, collation_fn=collate)
    
    return dataset
