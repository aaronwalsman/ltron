#!/usr/bin/env python
import os

import tqdm

from ltron.bricks.brick_scene import BrickScene
import ltron.settings as settings

#files = list(Path('./data/OMR/ldraw/').resolve().glob('*.mpd'))
omr_ldraw_path = os.path.join(settings.paths['omr'], 'ldraw')
files = list(os.listdir(omr_ldraw_path))
cleaned_files = []
for f in tqdm.tqdm(files):
    if f.startswith('.'):
        continue
    f = os.path.join(omr_ldraw_path, f)
    try:
        scene = BrickScene()
        scene.import_ldraw(str(f))
    except Exception as e:
        print(f"Failed to load {f}")
        continue
    instances = len(scene.instances.instances)
    unique = len(scene.shape_library.keys())
    cleaned_files.append((f, instances, unique))

cleaned_files.sort(key=lambda x: x[1], reverse=False)

for f, i, u in cleaned_files[:600]:
    print(f, ':', i, u)

# Chose the models where the size was under 100 after doing some manual hunting
subset = cleaned_files[:600]
# After this sort, can then subsample by number of unique pieces if desired
subset.sort(key=lambda x: x[2])

subsets = []

# Calling this again because didn't want to hold all the sets of piece in memory for the really big lego sets
for f, count, _ in tqdm(subset):
    try:
        scene = BrickScene()
        scene.import_ldraw(str(f))
    except Exception as e:
        print(f"Failed to load {f}")
        continue
    unique = set(scene.shape_library.keys())
    subsets.append((f, count, unique))

subsets = [s for s in subsets if len(s[2]) > 10]

def find_maximal_subset(lst, i, thresh):
    models = set()
    current_part_set = lst[i][2]
    while len(current_part_set) < thresh and len(models) < len(lst):
        minimal_size = 1000000
        min_union = set()
        idx = -1
        for count, elem in enumerate(lst):
            if count in models:
                continue
            else:
                union = current_part_set.union(elem[2])
                if len(union) < minimal_size:
                    minimal_size = len(union)
                    min_union = union
                    idx = count
        models.add(idx)
        current_part_set = min_union
    return models

maximal_subsets = [find_maximal_subset(subsets, i, 500) for i in range(len(subsets))]
maximal_subsets.sort(key=lambda x: len(x))
names = [subsets[idx][0] for idx in maximal_subsets[-1]]
data = {}
data['splits'] = {'all': []}
data['splits']['all'] = ['ldraw/' + f.name for f in names]
all_colors = set()
edges_per_scene = []
instances_per_scene = []
instance_counts = {}
instance_counts = {}
all_colors = set()
edges_per_scene = []
instances_per_scene = []
for f in tqdm(names):
    scene = BrickScene()
    scene.make_track_snaps()
    scene.import_ldraw(str(f))
    instances_per_scene.append(len(scene.instances))
    for instance_id, instance in scene.instances.items():
        brick_shape = instance.brick_shape
        if str(brick_shape) not in instance_counts:
            instance_counts[str(brick_shape)] = 0
        instance_counts[str(brick_shape)] += 1
        all_colors.add(instance.color)
    try:
        edges = scene.get_assembly_edges(unidirectional=True)
        edges_per_scene.append(edges.shape[1])
    except:
        print('poop')


data['max_instances_per_scene'] = max(instances_per_scene)
data['max_edges_per_scene'] = max(edges_per_scene)
data['shape_ids'] = dict(zip(sorted(instance_counts.keys()), range(1, len(instance_counts) + 1)))
data['all_colors'] = list(sorted(all_colors, key=int))

import json
with open('289_500_data.json', 'w') as f:
    json.dump(data, f)
