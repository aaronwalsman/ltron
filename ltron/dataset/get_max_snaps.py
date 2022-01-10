import tqdm

from ltron.dataset.paths import get_dataset_info
from ltron.bricks.brick_shape import BrickShape

def get_max_snaps(dataset):
    dataset_info = get_dataset_info(dataset)
    max_snaps = 0
    for brick_shape_name in tqdm.tqdm(dataset_info['shape_ids']):
        brick_shape = BrickShape(brick_shape_name)
        num_snaps = len(brick_shape.snaps)
        max_snaps = max(max_snaps, num_snaps)
    
    return max_snaps

if __name__ == '__main__':
    print(get_max_snaps('omr_split_4'))
