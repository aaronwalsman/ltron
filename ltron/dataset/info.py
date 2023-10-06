import json
import tarfile

import ltron.settings as settings
from ltron.exceptions import LtronMissingDatasetException

def get_dataset_info(dataset):
    try:
        return json.load(open(settings.DATASETS[dataset]))
    except KeyError:
        raise LtronMissingDatasetException(dataset)

def get_split_shards(dataset, split):
    info = get_dataset_info(dataset)
    return info['splits'][split]['shards']

def get_split_length(dataset, split):
    shards = get_split_shards(dataset, split)
    total = 0
    for shard in shards:
        shard_path = settings.SHARDS[shard]
        shard_file = tarfile.TarFile(shard_path)
        total += len(shard_file.getnames())
    
    return total
