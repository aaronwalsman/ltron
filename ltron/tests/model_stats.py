import tqdm

from ltron.bricks.brick_scene import BrickScene
from ltron.dataset.paths import get_dataset_paths

paths = get_dataset_paths('tiny_turbos3', 'all')
max_dim = 0
max_path = None
bounds = 400
too_big = []
iterate = tqdm.tqdm(paths['mpd'])
for path in iterate:
    scene = BrickScene()
    scene.import_ldraw(path)
    vmin, vmax = scene.get_bbox()
    offset = vmax - vmin
    max_offset = max(offset)
    if max_offset > max_dim:
        max_dim = max_offset
        max_path = path
    
    if max_offset > bounds:
        too_big.append(path)
    
    iterate.set_description('Max: %f'%max_dim)

print(max_dim)
print(max_path)
print('too big:')
for path in too_big:
    print(path)
