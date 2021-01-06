from brick_gym.geometry.grid_bucket import GridBucket
import brick_gym.bricks.snap as snap

class SnapManager:
    def __init__(self, cell_size=8):
        self.grid_bucket = GridBucket(cell_size)
        
    def insert_snap(self, snap_id, snap):
        
