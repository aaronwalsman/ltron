import numpy

def translate_instance(scene, instance, offset):
    transform = scene.instances[intance].transform
    transform[:3,3] += offset
    scene.move_instance(instance, transform)

def translate_instance_snapped(
    scene,
    instance,
    offset,
    step_size=(20,8,20),
):
    
    try:
        len(step_size)
    except TypeError:
        step_size = (step_size, step_size, step_size)
    
    magnitude = numpy.abs(offset)
    snapped_offset = None
    if magnitude[0] >= magnitude[1] and magnitude[0] >= magnitude[2]:
        snapped_offset = [step_size[0] * numpy.sign(offset[0]), 0, 0]
    elif magnitude[1] >= magnitude[2]:
        snapped_offset = [0, step_size[1] * numpy.sign(offset[1]), 0]
    else:
        snapped_offset = [0, 0, step_size[2] * numpy.sign(offset[2])]

def translate_instance_snapped_screen_space(
    scene,
    instance,
    offset,
    step_size=(20,8,20),
    min_2d_norm=0.1,
):
    
    assert scene.renderable
    
    try:
        len(step_size)
    except TypeError:
        step_size = (step_size, step_size, step_size)
    
    # get the view matrix (inverse of the camera pose)
    view_matrix = scene.get_view_matrix()
    
    # put the instance transform in camera space
    instance_transform = scene.instances[instance].transform
    instance_camera_space = view_matrix @ instance_transform
    
    # take each of the three local xyz axes in camera local space, normalize
    # them (as long as they are long enough) and then compare each with the
    # the desired 2d offset.
    screen_space_axes = instance_camera_space[:2,0:3]
    norm = numpy.sum(screen_space_axes**2, axis=0)**0.5
    
    best_dot = float('-inf')
    best_sign = 0
    best_axis = None
    
    for i in range(3):
        if norm[i] >= min_2d_norm:
            screen_space_axis = screen_space_axes[:,i] / norm[i]
            dot = screen_space_axis @ offset
            abs_dot = numpy.abs(dot)
            if abs_dot > best_dot:
                best_dot = abs_dot
                best_sign = numpy.sign(dot)
                best_axis = i
    
    # this should never happen unless min_2d_norm is really large
    assert best_axis is not None
    
    # apply the translation
    offset = [0,0,0]
    offset[best_axis] += step_size[best_axis] * best_sign
    
    instance_transform[:3,3] += offset
    scene.move_instance(instance, instance_transform)

def translate_instance_snapped_camera_space(
    scene,
    instance,
    offset,
    step_size=(20,8,20),
):
    
    assert scene.renderable
    
    try:
        len(step_size)
    except TypeError:
        step_size = (step_size, step_size, step_size)
        
    # get the view matrix (inverse of the camera pose)
    view_matrix = scene.get_view_matrix()
    
    # put the instance transform in camera space
    instance_transform = scene.instances[instance].transform
    instance_camera_space = view_matrix @ instance_transform
    
    # take each of the three local xyz axes in camera local space
    screen_space_axes = instance_camera_space[:2,0:3]
