from ltron.score import edit_distance

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class EditDistance(LtronGymComponent):
    def __init__(self,
        initial_assembly_component,
        current_assembly_component,
        shape_ids,
        miss_a_penalty=1,
        miss_b_penalty=2,
        pose_penalty=1,
    ):
        self.initial_assembly_component = initial_assembly_component
        self.current_assembly_component = current_assembly_component
        self.part_names = {value:key for key, value in shape_ids.items()}
        self.miss_a_penalty = miss_a_penalty
        self.miss_b_penalty = miss_b_penalty
        self.pose_penalty = pose_penalty

    def observe(self):
        initial_assembly = self.initial_assembly_component.observe()
        current_assembly = self.current_assembly_component.observe()
        
        self.edit_distance, _ = edit_distance(
            current_assembly,
            initial_assembly,
            self.part_names,
            miss_a_penalty=self.miss_a_penalty,
            miss_b_penalty=self.miss_b_penalty,
            pose_penalty=self.pose_penalty
        )

    def reset(self):
        self.observe()
        return None

    def step(self, action):
        self.observe()
        return None, -self.edit_distance, False, {}

    def set_state(self, state):
        self.observe()
        return None
