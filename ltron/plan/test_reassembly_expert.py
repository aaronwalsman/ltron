import random
import time
random.seed(1234)
from ltron.bricks.brick_scene import BrickScene
from ltron.experts.reassembly import ReassemblyExpert
from ltron.matching import match_configurations

scene = BrickScene(renderable=True, track_snaps=True)
scene.import_ldraw(
    #'~/.cache/ltron/collections/omr/ldraw/8661-1 - Carbon Star.mpd'
    #'~/.cache/ltron/collections/omr/ldraw/7657-1 - AT-ST.mpd'
    '~/.cache/ltron/collections/omr/ldraw/10030-1 - Imperial Star Destroyer - UCS.mpd'
)
print('loaded')

class_ids = {
    str(brick_type) : i
    for i, brick_type in enumerate(scene.brick_library.values(), start=1)
}

color_ids = {
    str(color) : i
    for i, color in enumerate(scene.color_library.values(), start=0)
}

target_config = scene.get_configuration(class_ids, color_ids)
#scene.remove_instance(23)
n = len(scene.instances)
remove = set(random.randint(1, n) for _ in range(10))
for r in remove:
    transform = scene.instances[r].transform.copy()
    transform[0,3] += random.randint(-5,5)*20
    transform[1,3] += random.randint(-5,5)*8
    transform[2,3] += random.randint(-5,5)*20
    scene.move_instance(r, transform)
workspace_config = scene.get_configuration(class_ids, color_ids)

t0 = time.time()
matches = match_configurations(workspace_config, target_config)
t1 = time.time()
print('t: %.06f'%(t1-t0))

'''
expert = ReassemblyExpert(1, class_ids)

target_config = scene.get_configuration(class_ids, color_ids)

scene.remove_instance(23)

workspace_config = scene.get_configuration(class_ids, color_ids)

expert.add_remove_brick(workspace_config, target_config)
'''
