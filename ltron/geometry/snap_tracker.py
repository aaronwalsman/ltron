import os
import time

import numpy

from aabbtree import AABB, AABBTree

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene

print('WARNING: EXPERIMENTAL')

class SnapTracker:
    def __init__(self):
        self.tree = AABBTree()
    
    def add_snap(self, snap):
        box = numpy.array(snap.bbox())
        box = AABB(box.T)
        self.tree.add(box, snap)
    
    def get_connections(self, snap):
        tbs = time.time()
        box = numpy.array(snap.bbox())
        tbe = time.time()
        box = AABB(box.T)
        t_start = time.time()
        print('bleep')
        potential_connections = self.tree.overlap_values(box)
        print('bloop')
        t_end = time.time()
        connections = []
        for other_snap in potential_connections:
            if snap.connected2(other_snap):
                connections.append(other_snap)
        
        return connections, t_end - t_start, tbe-tbs

def test():
    scene = BrickScene(track_snaps=True)
    path = os.path.join(
        settings.collections['omr'],
        #'ldraw/7657-1 - AT-ST.mpd')
        'ldraw/10030-1 - Imperial Star Destroyer - UCS.mpd')
    scene.import_ldraw(path)
    
    '''
    t0 = time.time()
    connections = scene.get_all_snap_connections()
    t1 = time.time()
    
    print('Traditional: %f'%(t1-t0))
    
    t_build_start = time.time()
    tracker = SnapTracker()
    k = 0
    for i, instance in scene.instances.items():
        for snap in instance.snaps:
            tracker.add_snap(snap)
            k += 1
    t_build_end = time.time()
    print('Build: %f'%(t_build_end - t_build_start))
    print('k: %i'%k)
    
    t = 0
    t_b = 0
    t2 = time.time()
    all_connections = {}
    for i, instance in scene.instances.items():
        for j, snap in enumerate(instance.snaps):
            connections, tt, ttt = tracker.get_connections(snap)
            t += tt
            t_b += ttt
            if connections:
                all_connections[i,j] = connections
    t3 = time.time()
    print('AABB: %f'%t)
    print('Box stuff: %f'%t_b)
    print('all: %f'%(t3-t2))
    
    import pdb
    pdb.set_trace()
    '''
    
    directions = {}
    s = 0
    for i, instance in scene.instances.items():
        for j, snap in enumerate(instance.snaps):
            s += 1
            direction = snap.transform[:3,1]
            direction /= numpy.linalg.norm(direction)
            direction = tuple(direction)
            for d in directions:
                if numpy.dot(d, direction) > 0.99:
                    directions[d].append(snap)
                    break
            else:
                directions[direction] = [snap]
    
    print('%i snaps'%s)
    print('%i directions:'%len(directions))
    for d in directions:
        print(d, ':', len(directions[d]))

if __name__ == '__main__':
    test()
