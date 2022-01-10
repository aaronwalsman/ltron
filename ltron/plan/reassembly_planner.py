import numpy

'''
Want:
An expert that can provide demonstrations of full assembly of a target scene,
while guaranteeing visual feasibility.

Desirable Properties:
Reduce camera motion as much as possible.
Don't take forever.

Zoey's idea:
Start from the full model.
Disassemble it.
Rebuild it in the disassembled order.

The trick is, we don't know where the "bricks on the ground" will be.  Maybe we should do bricks in the hand?  That would make that problem a lot easier.

The other thing is, we don't know what low-level actions to take even we know the brick order, but that's a much easier thing to plan out.
'''

class PlanningError(Exception):
    pass

def place_first_brick(
    
    target_transform
):
    pass

def move_brick_to_location(
    scene,
    target_instance,
    target_transform,
):
    
    steps = []
    
    # find edges in the target configuration
    original_transform = target_instance.transform
    scene.move_instance(target_instance, target_transform)
    target_edges = configuration.get_all_edges([new_brick])
    
    if not target_edges:
        raise PlanningError('No edges found in new configuration')
    
    # select one of the edges
    num_edges = target_edges.shape[1]
    i = random.randint(0, num_edges-1)
    instance_a, instance_b, snap_a, snap_b = num_edges[:,i]
    if instance_a == target_instance.instance_id:
        target_snap_id = snap_a
        destination_snap_id = snap_b
        destination_instance_id = instance_b
    elif instance_b == target_instance.instance_id:
        target_snap_id = snap_b
        destination_snap_id = snap_a
        destination_instance_id = instance_a
    else:
        raise PlanningError(
            'Edge does not involve target, this should never happen')
    
    # pick and place
    steps.append((
        'pick_and_place',
        target_snap_id,
        destination_instance_id,
        destination_snap_id,
    ))
    
    target_snap = target_instance.get_snap(target_snap_id)
    scene.move_instance(target_instance, original_transform)
    original_snap = target_instance.get_snap(target_snap_id)
    
    scene.pick_and_place_snap_transform(
        (target_instance.instance_id, target_snap_id),
        (destination_instance, destination_snap_id),
    )
    
    # rotate
    pnp_snap = target_instance.get_snap(target_snap_id)
    current_rotate = Quaternion(matrix=pnp_snap.transform)
    target_rotate = Quaternion(matrix=target_snap.transform)
    offset = current_rotate * target_rotate.inverse()
    if abs(offset.angle) < math.radians(2.5) :
        pass
    elif abs(offset.angle - math.radians( 90.)) < math.radians(2.5) :
        steps.append(('rotate', math.radians( 90.), target_snap_id))
    elif abs(offset.angle - math.radians(180.)) < math.radians(2.5) :
        steps.append(('rotate', math.radians(180.), target_snap_id))
    elif abs(offset.angle - math.radians(270.)) < math.radians(2.5) :
        steps.append(('rotate', math.radians(270.), target_snap_id))
    else:
        raise PlanningError('Unsupported offset required')
        
    return steps


def steps_from_target_and_brick_order(target_scene, brick_order):
    work_in_progress = BrickScene()
    
    steps = []
    
    first_instance_id = brick_order[0]
    first_instance = target_scene.instances[first_instance_id]
    steps.extend(place_first_brick(whatever, arguments, we, need, here))
    
    for next_brick_id in brick_order[1:]:
        steps.extend(move_brick_to_location(
            work_in_progress, next_instance, target_transform))
    
    return steps

def plan_reassembly(env):
    observation = env.reset()
    
    scene = env.components['scene'].brick_scene
    
    target_bricks = scene.components['reassembly'].target_bricks
    target_neighbors = scene.components['reassembly'].target_neighbors
    target_edges = scene.get_all_edges()
    
    # pick an arbitrary disassembly order
    visible_snaps = get_visible_bricks_and_snaps(
        observation['snap_pos'], observation['snap_neg'])
    brick_order = []
    while visible_snaps.shape[0] > 1:
        bricks_attempted_to_remove = set()
        removable_bricks = set()
        for instance_id, snap_id in visible_snaps:
            if instance_id in bricks_attempted_to_remove:
                continue
            
            bricks_attempted_to_remove.add(instance_id)
            
            snap = scene.instances[instance_id].get_snap(snap_id)
            
            # check push
            #for direction in 'pull', 'push':
            movable = scene.check_snap_collision([instance_id], snap)
            if movable:
                removable_bricks.add((instance_id, snap_id, direction))
                break
        
        instance_id, snap_id, direction = random.choice(list(removable_bricks))
        brick_order.append(instance_id)
        
        action = reassembly_template_action()
        action['disassembly'] = {
            'activate' : True,
            'polarity' : '-+'.index(polarity),
            'direction' : ('pull', 'push').index(direction),
            'pick' : (pick_y, pick_x),
        }
        observation, reward, terminal, info = env.step(action)
        visible_snaps = get_visible_bricks_and_snaps(
            observation['snap_pos'], observation['snap_neg'])
    
    brick_order.reverse()

def get_visible_snaps(pos_snaps, neg_snaps):
    pos_snaps = pos_snaps.reshape(-1,2)
    neg_snaps = neg_snaps.reshape(-1,2)
    snaps = numpy.concatenate((pos_snaps, neg_snaps), axis=0)
    
    unique_snaps = numpy.unique(snaps, axis=0)
    
    return unique_snaps

def get_all_viewpoints(env):
    viewpoint_component = scene.components['viewpoint']
    start_position = viewpoint_component.position
    

def set_configuration(scene, bricks):
    scene.clear_instances()
    for brick in bricks:
        scene.add_instance(brick.brick_shape, brick.color, brick.transform)


