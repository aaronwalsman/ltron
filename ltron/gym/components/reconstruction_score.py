from ltron.gym.components.ltron_gym_component import LtronGymComponent

class ReconstructionScore(LtronGymComponent):
    def __init__(self,
        scene_component,
    ):
        self.scene_component = scene_component
    
    def reset(self):
        scene = self.scene_component.scene
        target_bricks, target_neighbors = scene.get_brick_neighbors()
        target_bricks = [brick.clone() for brick in target_bricks]
        target_neighbors = [
            [neighbor.clone() for neighbor in brick_neighbors]
            for brick_neighbors in target_neighbors
        ]
        
        return None
    
    def step(self, action):
        scene = self.scene_component.scene
        current_bricks, current_neighbors = scene.get_brick_neighbors()
        score = score_configurations(
            target_bricks, target_neighbors, current_bricks, current_neighbors)
        
        return None, score, False, {}
