import math
import itertools

import numpy

def best_first_total_alignment(scene):
    
    # build a list of initial islands
    islands = [
        set([instance_id])
        for instance_id, instance in scene.instances.items()
        #if len(instance.get_snaps())
    ]
    #existing_connections = scene.get_all_snap_connections()
    #consumed_instances = set()
    #for instance_id in scene.instances:
    #    if instance_id in consumed_instances:
    #        continue
    #    island = set([instance_id])
    #    for other_id, _, _ in existing_connections.get(instance_id, []):
    #        island.add(other_id)
    #        consumed_instances.add(other_id)
    #    islands.append(set([instance_id]))
    
    def distance_between_islands(island_i, island_j, alignment_factor=50):
        best_snaps = None
        best_distance = float('inf')
        for instance_i in island_i:
            instance_i = scene.instances[instance_i]
            for i, snap_i in enumerate(instance_i.get_snaps()):
                position_i = numpy.dot(snap_i.transform, [0,0,0,1])[:3]
                axis_i = numpy.dot(snap_i.transform, [0,1,0,0])[:3]
                for instance_j in island_j:
                    instance_j = scene.instances[instance_j]
                    for j, snap_j in enumerate(instance_j.get_snaps()):
                        position_j = numpy.dot(snap_j.transform, [0,0,0,1])[:3]
                        axis_j = numpy.dot(snap_j.transform, [0,1,0,0])[:3]
                        distance = numpy.linalg.norm(position_i - position_j)
                        alignment = abs(numpy.dot(axis_i, axis_j))
                        surrogate_distance = (
                            distance + (1. - alignment) * alignment_factor)
                        if (surrogate_distance < best_distance and
                            snap_i.connected(
                                snap_j,
                                alignment_tolerance=-float('inf'),
                                distance_tolerance=float('inf'))
                        ):
                            best_distance = distance
                            best_snaps = ((instance_i, i), (instance_j, j))
        
        return best_snaps, best_distance
    
    step = 0
    while len(islands) > 1:
        step += 1
        best_islands = None
        best_snaps = None
        best_distance = float('inf')
        for i, j in itertools.combinations(range(len(islands)), 2):
            island_i = islands[i]
            island_j = islands[j]
            
            ij_snaps, ij_distance = distance_between_islands(island_i, island_j)
            if ij_distance < best_distance:
                best_distance = ij_distance
                best_snaps = ij_snaps
                best_islands = (i,j)
        
        (instance_i, i), (instance_j, j) = best_snaps
        
        snap_i = scene.instances[instance_i].get_snap(i)
        snap_j = scene.instances[instance_j].get_snap(j)
        
        best_offset = best_alignment(snap_i.transform, snap_j.transform)
        
        # move the islands together
        island_j = islands[best_islands[-1]]
        for inst_j in island_j:
            existing_transform = scene.instances[inst_j].transform
            new_transform = best_offset @ existing_transform
            scene.set_instance_transform(inst_j, new_transform)
        
        # merge the islands
        i,j = best_islands
        #print('merging')
        #print(islands[i])
        #print(islands[j])
        #scene.export_ldraw('./reconstruction/step_%i.mpd'%step)
        islands = [
            island for k, island in enumerate(islands)
            if k != i and k != j] + [islands[i] | islands[j]]

def best_alignment(transform_a, transform_b):
    #rotation_a = transform_a[:3, :3]
    #rotation_b = transform_b[:3, :3]
    if numpy.linalg.det(transform_a[:3,:3]) < 0.:
        transform_a[0:3,0] *= -1.
    if numpy.linalg.det(transform_b[:3,:3]) < 0.:
        transform_b[0:3,0] *= -1.
    best_offset = None
    best_angle = -float('inf')
    for i in range(4):
        theta = i * math.pi/2.
        c = math.cos(theta)
        s = math.sin(theta)
        r = numpy.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]])
        spin_offset = transform_a @ r @ numpy.linalg.inv(transform_b)
        surrogate_angle = numpy.trace(spin_offset[:3,:3])
        if surrogate_angle > best_angle:
            best_offset = spin_offset
            best_angle = surrogate_angle
    
    return best_offset
