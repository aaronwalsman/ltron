from brick_gym.geometry.grid_bucket import GridBucket
import brick_gym.ldraw.snap as snap

class SnapManager:
    def __init__(self, cell_size=8):
        self.grid_bucket = GridBucket(cell_size)
    
    def insert_snap(self, snap_id, snap_position):
        self.grid_bucket.insert(snap_id, snap_position)
    
    def insert_brick(self, 
