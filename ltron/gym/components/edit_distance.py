from ltron.score import edit_distance

from ltron.gym.components.ltron_gym_component import LtronGymComponent


class EditDistance(LtronGymComponent):
    def __init__(self,
        initial_assembly_component,
        current_assembly_component,
        shape_ids,
    ):
        self.initial_assembly_component = initial_assembly_component
        self.current_assembly_component = current_assembly_component
        self.part_names = {value:key for key, value in shape_ids.items()}

    def observe(self):
        initial_assembly = self.initial_assembly_component.assembly
        current_assembly = self.current_assembly_component.assembly
        
        self.edit_distance, _ = edit_distance(
            current_assembly,
            initial_assembly,
            self.part_names,
            miss_b_penalty=2,
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

