import random
import math

from gym.spaces import Dict, Discrete

from pyquaternion import Quaternion

from ltron.gym.spaces import BrickShapeColorSpace, BrickShapeColorPoseSpace

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class RandomBrickInserter(LtronGymComponent):
    def __init__(
        self,
        scene_component,
        shape_ids,
        color_ids,
        insert_frequency='reset',
        transform=None,
        clear_on_insert=True,
        randomize_orientation=False,
    ):
        self.scene_component = scene_component
        self.brick_shape_to_id = shape_ids
        self.brick_color_to_id = color_ids
        self.insert_frequency = insert_frequency
        if transform is None:
            scene = self.scene_component.brick_scene
            transform = scene.upright
        self.transform = transform
        self.clear_on_insert = clear_on_insert
        self.randomize_orientation = randomize_orientation
    
    def reset(self):
        if self.insert_frequency == 'reset' or self.insert_frequency == 'step':
            self.insert_brick()
        
        return None
    
    def step(self, action):
        if self.insert_frequency == 'step':
            self.insert_brick()
        return None, 0., False, {}
    
    def insert_brick(self):
        shape = random.choice(list(self.brick_shape_to_id.keys()))
        color = random.choice(list(self.brick_color_to_id.keys()))
        scene = self.scene_component.brick_scene
        if self.clear_on_insert:
            scene.clear_instances()
        if self.randomize_orientation:
            orientation = Quaternion.random().transformation_matrix
            # TMP
            #orientation = Quaternion.random()
            #random_axis = orientation.rotation_matrix @ [0,1,0]
            #random_angle = random.random() * math.radians(20)
            #orientation = Quaternion(
            #    axis=random_axis, angle=random_angle).transformation_matrix
            # TMP
            transform = self.transform @ orientation
        else:
            transform = self.transform
        scene.add_instance(shape, color, transform)

class BrickInserter(LtronGymComponent):
    def __init__(
        self,
        scene_component,
        shape_ids,
        color_ids,
        include_pose=False,
        clear_scene=True,
    ):
        self.scene_component = scene_component
        self.brick_shape_to_id = shape_ids
        self.id_to_brick_shape = {value:key for key, value in shape_ids.items()}
        self.color_name_to_id = color_ids
        self.id_to_color_name = {value:key for key, value in color_ids.items()}
        self.include_pose = include_pose
        self.clear_scene = clear_scene
        
        if self.include_pose:
            #self.action_space = Dict({
            #    'shape_color' : BrickShapeColorSpace(shape_ids, color_ids),
            #    'pose' : SE3Space(),
            #})
            self.action_space = BrickShapeColorPoseSpace(shape_ids, color_ids)
        else:
            self.action_space = BrickShapeColorSpace(shape_ids, color_ids)
    
    def step(self, action):
        shape = action[0]
        color = action[1]
        
        if shape == 0 or color == 0:
            success = False
        elif (
            shape in self.id_to_brick_shape and
            color in self.id_to_color_name
        ):
            scene = self.scene_component.brick_scene
            if self.clear_scene:
                scene.clear_instances()
            brick_shape = self.id_to_brick_shape[shape]
            color_name = self.id_to_color_name[color]
            scene.add_instance(brick_shape, color_name, scene.upright)
            success = True
        else:
            success = False
        
        return None, 0., False, {}
    
    def no_op_action(self):
        return (0,0)
    
    def actions_to_insert_brick(self, shape, color, pose=None):
        if self.include_pose:
            return shape, color, pose
        else:
            return shape, color

class HandspaceBrickInserter(LtronGymComponent):
    def __init__(
        self,
        hand_component,
        table_component,
        shape_ids,
        color_ids,
        max_instances
    ):
        self.hand_component = hand_component
        self.table_component = table_component
        self.brick_shape_to_id = shape_ids
        self.id_to_brick_shape = {value:key for key, value in shape_ids.items()}
        self.color_name_to_id = color_ids
        self.id_to_color_name = {value:key for key, value in color_ids.items()}
        self.max_instances = max_instances
        
        self.observation_space = Dict({'success' : Discrete(2)})
        self.action_space = Dict({
            'shape' : Discrete(max(self.id_to_brick_shape.keys())+1),
            'color' : Discrete(max(self.id_to_color_name.keys())+1),
        })
    
    def reset(self):
        return {'success' : False}
    
    def step(self, action):
        if len(self.table_component.brick_scene.instances):
            num_instances = max(
                self.table_component.brick_scene.instances.keys())
        else:
            num_instances = 0
        if num_instances > self.max_instances:
            success = False
        elif (action['shape'] in self.id_to_brick_shape and
            action['color'] in self.id_to_color_name
        ):
            scene = self.hand_component.brick_scene
            scene.clear_instances()
            brick_shape = self.id_to_brick_shape[action['shape']]
            color_name = self.id_to_color_name[action['color']]
            scene.add_instance(brick_shape, color_name, scene.upright)
            success = True
        else:
            success = False
        
        return {'success' : success}, 0, False, {}
    
    def no_op_action(self):
        return {'shape':0, 'color':0}
