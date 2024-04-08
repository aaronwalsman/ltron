import time
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import build_collision_map

scene = BrickScene(track_snaps=True, renderable=True, collision_checker=True)
scene.import_ldraw('./model2_no_wheels.mpd')

t0 = time.time()
a = scene.get_assembly()
t1 = time.time()
print('Assembly: %f'%(t1-t0))

t2 = time.time()
cm = build_collision_map(scene)
t3 = time.time()
print('Collision Map: %f'%(t3-t2))

breakpoint()
