import os
import json

import tqdm

import ltron.settings as settings
from ltron.dataset.paths import (
        get_dataset_paths, get_dataset_info, get_metadata_path)
import ltron.ldraw.documents as documents

def make_dataset_metadata(dataset, split):
    directory = settings.datasets[dataset]
    dataset_info = get_dataset_info(dataset)
    shape_ids = dataset_info['shape_ids']
    file_paths = get_dataset_paths(dataset, split)
    for file_path in tqdm.tqdm(file_paths):
        document = documents.LDrawDocument.parse_document(file_path)
        parts = document.get_all_parts()
        metadata = {
            'class_labels' : {i+1 : shape_ids[part[0]]
                for i, part in enumerate(parts)}
        }
        
        # temp hack that only works for random_stack edges
        metadata['edges'] = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line_parts = line.split()
                if (len(line_parts) >= 3 and
                        line_parts[0] == '0' and
                        line_parts[1] == 'EDGE'):
                    a, b = line_parts[2].split(',')
                    a = int(a)+1
                    b = int(b)+1
                    metadata['edges'].append((a,b))
        
        metadata_path = get_metadata_path(file_path)
        json.dump(metadata, open(metadata_path, 'w'))
