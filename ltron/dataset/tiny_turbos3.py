#!/usr/bin/env python
import os
import random
import json

import tqdm

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene

breakout = {
    '4096 - Micro Wheels - AB Truck and Trailer.mpd' : [
            '4096 - ab truck.ldr'
    ],
    '8123' : ['8123 - 1.ldr'],
            #'8123 - 2.ldr'], # duplicate of - 1 except for button helmet color
    '8124' : ['8124 - 1.ldr',
            '8124 - 2.ldr'],
    '8125' : ['8125 - 8125-1.ldr',
            '8125 - 8125-2.ldr'],
    '8126' : ['8126 - 8126-1.ldr',
            '8126 - 8126-2.ldr'],
    '8134' : ['8134 - 8134-1.ldr',
            '8134 - 8134-2.ldr',
            #'8134 - 8134-3.ldr', # 164 parts, second largest
            '8134 - 8134-3-a.ldr', # just the cab please
            '8134 - 8134-3-b.ldr', # and actualy the flatbed too
    ],
    '8135' : ['8135 - 8135-1.ldr',
            '8135 - 8135-2.ldr',
            '8135 - 8135-3.ldr',
            '8135 - 8135-4.ldr'],
    '8147' : ['8147 - 8147-1.ldr',
            '8147 - 8147-2.ldr',
            '8147 - 8147-3.ldr', # 104 parts, seems manageable
            #'8147 - 8147-4.ldr', # 101 parts, but longer than 8147-3
            '8147 - 8147-4-a.ldr', # cab
            '8147 - 8147-4-c.ldr', # trailer
            '8147 - 8147-5.ldr',
            '8147 - 8147-6.ldr',
            #'8147 - 8147-7.ldr'], # 121 parts, quite large
            '8147 - 8147-7-a.ldr', # but hey!  we can separate it out!
            #'8147 - 8147-7-b.ldr', # just a connector
            #'8147 - 8147-7-c.ldr', # no wheels, long and has strange pieces
            '8147 - 8147-7-d.ldr', # end of the trailer
    ],
    '8152' : ['8152 - 8152-1.ldr',
            '8152 - 8152-2.ldr',
            '8152 - 8152-3.ldr',
            '8152 - 8152-4.ldr',
            '8152 - 8152-5.ldr'],
    '8154' : ['8154 - 8154-1.ldr',
            '8154 - 8154-2.ldr',
            '8154 - 8154-3.ldr',
            '8154 - 8154-4.ldr',
            '8154 - 8154-5.ldr',
            '8154 - 8154-6.ldr',
            #'8154 - 8154-7.ldr', # 202 parts, largest (way too big for 256x256)
            '8154 - 8154-7-a.ldr', # cab
            #'8154 - 8154-7-b.ldr' # trailer, long weird parts, quite large
            '8154 - 8154-8.ldr'],
    '8182' : ['8182 - 8182-1.ldr',
            '8182 - 8182-2.ldr',
            '8182 - 8182-3.ldr',
            '8182 - 8182-4.ldr',
            '8182 - 8182-5.ldr'],
    '8186' : ['8186 - 8186-1.ldr',
            '8186 - 8186-2.ldr',
            '8186 - 8186-3.ldr',
            '8186 - 8186-4.ldr',
            '8186 - 8186-5.ldr',
            '8186 - 8186-6.ldr',
            '8186 - 8186-7.ldr'],
    '8196' : ['8196 - 8196-1.ldr',
            '8196 - 8196-2.ldr'],
    '8197' : ['8197 - 8197-1.ldr',
            '8197 - 8197-2.ldr'],
    '8198' : ['8198 - 8198-1.ldr',
            '8198 - 8198-2.ldr'],
    '8199' : ['8199 - 8199-1.ldr',
            '8199 - 8199-2.ldr'],
    '8211' : ['8211 - 8211-1.ldr',
            '8211 - 8211-2.ldr',
            '8211 - 8211-3.ldr',
            '8211 - 8211-4.ldr',
            '8211 - 8211-5.ldr'],
    '8681' : ['8681 - 8681-1.ldr',
            '8681 - 8681-2.ldr',
            '8681 - 8681-3.ldr',
            '8681 - 8681-4.ldr',
            '8681 - 8681-5.ldr'],
}

# this is a list of tiny turbos sets according to bricklink
# check and see which ones exist in the OMR

# commented out entries have multiple models inside them
# these should probably be manually inspected to see if they have usable
# subdocuments once we have a way to include subdocuments in a dataset
set_numbers = [
'4096',
'4947',
'4948',
'4949',
'6111',
'7452',
'7453',
'7611',
'7612',
'7613',
'7800',
'7801',
'7802',
'8119',
'8120',
'8121',
'8122',
'8123', # multiple models
'8124', # multiple models
'8125', # multiple models
'8126', # multiple models
'8130',
'8131',
'8132',
'8133',
'8134', # multiple models
'8135', # multiple models
'8147', # multiple models
'8148',
'8149',
'8150',
'8151',
'8152', # multiple models
'8153',
'8154', # multiple models
#'8155', # multiple models but doesn't exist in OMR
'8182', # multiple models
'8186', # multiple models
'8192',
'8193',
'8194',
'8195',
'8196', # multiple models
'8197', # multiple models
'8198', # multiple models
'8199', # multiple models
'8211', # multiple models
'8301',
'8302',
'8303',
'8304',
'8595',
'8641',
'8642',
'8643',
'8644',
'8655',
'8656',
'8657',
'8658',
'8661',
'8662',
'8663',
'8664',
'8665',
'8666',
'8681', # multiple models
'30030',
'30033',
'30034',
'30035',
'30036']

existing_sets = {}
tiny_turbos3_path = settings.datasets['tiny_turbos3']
omr_ldraw = os.path.join(os.path.dirname(tiny_turbos3_path), 'ldraw')
all_sets = sorted(os.listdir(omr_ldraw))
for set_number in set_numbers:
    for subset_number in range(1,10):
        for set_name in all_sets:
            if subset_number == 1 and set_name.startswith(set_number + ' '):
                if set_number not in existing_sets:
                    existing_sets[set_number] = []
                existing_sets[set_number].append(set_name)
            elif set_name.startswith(set_number + '-' + str(subset_number)):
                if set_number not in existing_sets:
                    existing_sets[set_number] = []
                existing_sets[set_number].append(set_name)

# manual hack to fix "Black Racer.mpd" which has no set number in the file name
#breakout_paths.append(os.path.join(omr_ldraw, 'Black Racer.mpd'))
existing_sets['Black Racer'] = ['Black Racer.mpd']

print('%i sets found'%len(sum(existing_sets.values(), [])))

breakout_paths = []
scene = BrickScene()
scene.make_track_snaps()
instance_counts = {}
instances_per_scene = []
edges_per_scene = []
all_colors = set()
set_signatures = {}
for set_number, set_list in existing_sets.items():
    
    for existing_set in set_list:
        if set_number in breakout:
            file_paths = ['%s:%s'%(existing_set, subdocument)
                    for subdocument in breakout[set_number]]
        elif existing_set in breakout:
            file_paths = ['%s:%s'%(existing_set, subdocument)
                    for subdocument in breakout[existing_set]]
        else:
            file_paths = [existing_set]
        #breakout_paths.extend(file_paths)
        
        for file_path in file_paths:
            scene.clear_instances()
            scene.clear_assets()
            scene.import_ldraw(os.path.join(omr_ldraw, file_path))
            instances_per_scene.append(len(scene.instances))
            set_signature = {}
            for instance_id, instance in scene.instances.items():
                brick_shape = instance.brick_shape
                if str(brick_shape) not in instance_counts:
                    instance_counts[str(brick_shape)] = 0
                instance_counts[str(brick_shape)] += 1
                all_colors.add(instance.color)
                
                if str(brick_shape) not in set_signature:
                    set_signature[brick_shape] = 0
                set_signature[brick_shape] += 1
            set_signature = ','.join('%s:%i'%(key, value)
                    for key, value in set_signature.items())
            if set_signature not in set_signatures:
                breakout_paths.append(file_path)
                set_signatures[set_signature] = []
            set_signatures[set_signature].append(file_path)
            
            edges = scene.get_assembly_edges(unidirectional=True)
            edges_per_scene.append(edges.shape[1])
            print('%s:'%file_path)
            print('  %i instances'%len(scene.instances))
            print('  %i edges'%(edges.shape[1]))

print('%i broken-out sets found'%len(breakout_paths))
print('%i unique sets found'%len(set_signatures))
for set_signature, file_paths in set_signatures.items():
    if len(file_paths) > 1:
        print('Warning possible duplicated sets:')
        for file_path in file_paths:
            print(' ', file_path)

print('Average instances per model: %f'%(
        sum(instances_per_scene)/len(instances_per_scene)))
print('Min/Max instances per model: %i, %i'%(
        min(instances_per_scene), max(instances_per_scene)))

print('Average edges per model: %f'%(
        sum(edges_per_scene)/len(edges_per_scene)))
print('Min/Max edges per model: %i, %i'%(
        min(edges_per_scene), max(edges_per_scene)))

sorted_instance_counts = reversed(sorted(
        (value, key) for key, value in instance_counts.items()))

print('Part usage statistics:')
for count, brick_shape in sorted_instance_counts:
    print('%s: %i'%(brick_shape, count))

print('%i total brick shapes'%len(instance_counts))

random.seed(1234)
breakout_paths = list(sorted(breakout_paths))
test_set = sorted(random.sample(breakout_paths, 20))
train_set = [path for path in breakout_paths if path not in test_set]

all_tiny_turbos = ['ldraw/' + set_name for set_name in breakout_paths]
train_tiny_turbos = ['ldraw/' + set_name for set_name in train_set]
test_tiny_turbos = ['ldraw/' + set_name for set_name in test_set]
dataset_info = {
    'splits' : {
        'all' : all_tiny_turbos,
        'train' : train_tiny_turbos,
        'test' : test_tiny_turbos
    },
    'max_instances_per_scene' : max(instances_per_scene),
    'max_edges_per_scene' : max(edges_per_scene),
    'shape_ids':dict(
            zip(sorted(instance_counts.keys()),
            range(1, len(instance_counts)+1))),
    'all_colors':list(sorted(all_colors, key=int))
}

with open(tiny_turbos3_path, 'w') as f:
    json.dump(dataset_info, f, indent=4)
