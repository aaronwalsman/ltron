#!/usr/bin/env python
from ltron.score import score_configurations, pseudo_f1
from ltron.bricks.brick_scene import BrickScene

duck_x = BrickScene(track_snaps=True)
duck_x.import_ldraw('./duck_x.mpd')

duck_y = BrickScene(track_snaps=True)
duck_y.import_ldraw('./duck_y.mpd')

def get_brick_neighbors(scene):
    edges = scene.get_assembly_edges()
    neighbor_ids = {}
    for i in range(edges.shape[1]):
        instance_a, instance_b, snap_a, snap_b = edges[:,i]
        if instance_a not in neighbor_ids:
            neighbor_ids[instance_a] = set()
        if instance_b not in neighbor_ids:
            neighbor_ids[instance_b] = set()
        neighbor_ids[instance_a].add(instance_b)
        neighbor_ids[instance_b].add(instance_a)
    
    brick_order = list(sorted(neighbor_ids.keys()))
    bricks = [scene.instances[brick_id] for brick_id in brick_order]
    neighbors = [
        [scene.instances[n_id] for n_id in neighbor_ids[brick_id]]
        for brick_id in brick_order
    ]
    
    return bricks, neighbors

x_bricks, x_neighbors = get_brick_neighbors(duck_x)
y_bricks, y_neighbors = get_brick_neighbors(duck_y)

x_scores, y_scores = score_configurations(
    x_bricks, x_neighbors, y_bricks, y_neighbors)

print('x scores:')
print(x_scores)

print('y scores:')
print(y_scores)

print('f1: %.04f'%pseudo_f1(x_scores, y_scores))

import pdb
pdb.set_trace()
