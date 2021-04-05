#!/usr/bin/env python
import math
import os
import json

import matplotlib.pyplot as pyplot

import brick_gym.config as config

output_path = os.path.join(config.paths['data'], 'omr_statistics')
if not os.path.exists(output_path):
    os.makedirs(output_path)

omr_composition_path = os.path.join(
        config.paths['data'], 'omr_composition.json')
with open(omr_composition_path, 'r') as f:
    composition = json.load(f)

all_types = set(
        sum([list(value.keys()) for value in composition.values()], []))
print('Number of part types in all sets: %i'%len(all_types))

type_counts = {key:len(value) for key, value in composition.items()}

pyplot.hist(type_counts.values(), bins=20, rwidth=0.8)
pyplot.savefig(os.path.join(output_path, 'type_hist.png'))
pyplot.clf()

instance_counts = {
        key:sum(value.values()) for key, value in composition.items()}

pyplot.hist(instance_counts.values(), bins=50, rwidth=0.8)
pyplot.savefig(os.path.join(output_path, 'instance_hist.png'))
pyplot.clf()

clipped_instance_counts_3k = {
        key:value for key, value in instance_counts.items() if value < 3000}

pyplot.hist(clipped_instance_counts_3k.values(), bins=50, rwidth=0.8)
pyplot.savefig(os.path.join(output_path, 'clipped_instance_hist_3k.png'))
pyplot.clf()

part_usage = {}
for model, data in composition.items():
    for part_name, count in data.items():
        try:
            part_usage[part_name] += count
        except:
            part_usage[part_name] = count
raw_part_usage = list(reversed(sorted(part_usage.values())))
fig, ax = pyplot.subplots()
ax.set_yscale('log')
pyplot.bar(range(len(raw_part_usage)), raw_part_usage)
pyplot.savefig(os.path.join(output_path, 'part_usage.png'))
pyplot.clf()

fig, ax = pyplot.subplots()
ax.set_yscale('log')
pyplot.bar(range(100), raw_part_usage[:100])
pyplot.savefig(os.path.join(output_path, 'part_usage_100.png'))
pyplot.clf()

clipped_instance_counts_1c = {
        key:value for key, value in instance_counts.items() if value < 100}

# >= 1000
one_thousand_or_more = {
        key for key, value in instance_counts.items() if value >= 1000}
print('Number of sets with >= 1000 instances: %i'%len(one_thousand_or_more))

# <= 100
print('Number of sets with <= 100 instances: %i'%len(
        clipped_instance_counts_1c))

small_composition = {
        key:value for key, value in composition.items()
        if key in clipped_instance_counts_1c}

small_types = set(
        sum([list(value.keys()) for value in small_composition.values()], []))
print('Number of parts in <= 100 instance sets: %i'%len(small_types))
