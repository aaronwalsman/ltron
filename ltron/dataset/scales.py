import tqdm

import ltron.dataset.paths as paths
from ltron.bricks.brick_scene import BrickScene

pico_max_bricks = 2
nano_max_bricks = 4
micro_max_bricks = 8
mini_max_bricks = 32
small_max_bricks = 128
medium_max_bricks = 512

def get_model_sizes(dataset, split, max_bricks=None):
    file_paths = paths.get_dataset_paths(dataset, split)
    bboxes = {}
    max_dimensions = {}
    bs = BrickScene()
    for path in tqdm.tqdm(file_paths['mpd']):
        bs.instances.clear()
        bs.import_ldraw(path)
        if max_bricks is None or len(bs.instances) <= max_bricks:
            bbox_min, bbox_max = bs.get_scene_bbox()
            bbox_dimensions = bbox_max - bbox_min
            bboxes[path] = bbox_dimensions
            max_dimensions[path] = max(bbox_dimensions)
    
    max_dim, max_path = max(zip(max_dimensions.values(), max_dimensions.keys()))
    avg_dim = sum(max_dimensions.values())/len(max_dimensions)
    
    print(max_dim, max_path)
    print(avg_dim)
    print(len([d for d in max_dimensions.values() if d < 480]),
        '/', len(max_dimensions))
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    get_model_sizes('omr', 'all', 128)
