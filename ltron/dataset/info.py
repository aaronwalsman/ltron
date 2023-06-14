import os
import json

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
