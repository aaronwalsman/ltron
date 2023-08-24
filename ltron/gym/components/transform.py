from gymnasium.spaces import Discrete

from supermecha import SuperMechaComponent

class TransformSnapComponent(SuperMechaComponent):
    def __init__(
        self,
        scene_component,
        check_collision,
        transforms,
        space='local',
    ):
        self.scene_component = scene_component
        self.check_collision = check_collision
        self.transforms = transforms
        self.space = space
        self.action_space = Discrete(len(transforms))

    def transform_snap(self, instance_id, snap_id, action):
        if instance_id == 0:
            return False

        transform = self.transforms[action]
        
        scene = self.scene_component.brick_scene
        instance = scene.instances[instance_id]
        if snap_id >= len(instance.snaps):
            return False
        snap = instance.snaps[snap_id]

        avoided_collision = scene.transform_about_snap(
            [instance],
            snap,
            transform,
            check_collision=self.check_collision,
            space=self.space,
        )
        
        return avoided_collision
