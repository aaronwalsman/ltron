#!/usr/bin/env python
import os
import random
import json

import tqdm

import brick_gym.config as config

# this is a list of tiny turbos sets according to bricklink
# check and see which ones exist in the OMR

# commented out entries have multiple models inside them
# these should probably be manually inspected to see if they have usable
# subdocuments once we have a way to include subdocuments in a dataset
set_numbers = [
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
############
#'8123',
#'8124',
#'8125',
#'8126',
###########
'8130',
'8131',
'8132',
'8133',
#'8134', # multiple models
#'8135',
#'8147',
'8148',
'8149',
'8150',
'8151',
#'8152', # multiple models
'8153',
###########
#'8154',
#'8155',
#'8182',
#'8186',
###########
'8192',
'8193',
'8194',
'8195',
###########
#'8196',
#'8197',
#'8198',
#'8199',
#'8211',
###########
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
###########
#'8681',
###########
'30030',
'30033',
'30034',
'30035',
'30036']

existing_sets = set()
tiny_turbos_path = config.datasets['tiny_turbos']
omr_ldraw = os.path.join(os.path.dirname(tiny_turbos_path), 'ldraw')
all_sets = sorted(os.listdir(omr_ldraw))
for set_number in set_numbers:
    for subset_number in range(1,10):
        for set_name in all_sets:
            if set_name.startswith(set_number + '-' + str(subset_number)):
                existing_sets.add(set_name)

print('%i sets found'%len(existing_sets))

from brick_gym.bricks.brick_scene import BrickScene
scene = BrickScene()
instance_counts = {}
instances_per_scene = []
for existing_set in existing_sets:
    scene.clear_instances()
    scene.clear_assets()
    scene.import_ldraw(os.path.join(omr_ldraw, existing_set))
    instances_per_scene.append(len(scene.instances))
    print('%s has %i instances'%(existing_set, len(scene.instances)))
    for instance_id, instance in scene.instances.items():
        brick_type = instance.brick_type
        if str(brick_type) not in instance_counts:
            instance_counts[str(brick_type)] = 0
        instance_counts[str(brick_type)] += 1

print('Average instances per model: %f'%(
        sum(instances_per_scene)/len(instances_per_scene)))
print('Min/Max instances per model: %i, %i'%(
        min(instances_per_scene), max(instances_per_scene)))

sorted_instance_counts = reversed(sorted(
        (value, key) for key, value in instance_counts.items()))

print('Part usage statistics:')
for count, brick_type in sorted_instance_counts:
    print('%s: %i'%(brick_type, count))

print('%i total brick types'%len(instance_counts))

random.seed(1234)
test_set = random.sample(existing_sets, int(len(existing_sets) * 0.25))
train_set = existing_sets - set(test_set)

all_tiny_turbos = ['ldraw/' + set_name for set_name in existing_sets]
train_tiny_turbos = ['ldraw/' + set_name for set_name in train_set]
test_tiny_turbos = ['ldraw/' + set_name for set_name in test_set]
dataset_info = {
    'splits' : {
        'all' : all_tiny_turbos,
        'train' : train_tiny_turbos,
        'test' : test_tiny_turbos
    },
    'max_instances_per_scene' : max(instances_per_scene),
    'class_ids':dict(
            zip(sorted(instance_counts.keys()),
            range(1, len(instance_counts)+1)))
}

with open(tiny_turbos_path, 'w') as f:
    json.dump(dataset_info, f, indent=4)
