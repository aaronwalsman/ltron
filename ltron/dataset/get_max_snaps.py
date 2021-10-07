import tqdm

from ltron.dataset.paths import get_dataset_info
from ltron.bricks.brick_type import BrickType

def get_max_snaps(dataset):
    dataset_info = get_dataset_info(dataset)
    max_snaps = 0
    for brick_type_name in tqdm.tqdm(dataset_info['class_ids']):
        brick_type = BrickType(brick_type_name)
        num_snaps = len(brick_type.snaps)
        max_snaps = max(max_snaps, num_snaps)
    
    return max_snaps

if __name__ == '__main__':
    print(get_max_snaps('omr_split_4'))
