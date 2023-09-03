import numpy

from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.snap import StudHole
from ltron.geometry.utils import translate_matrix
import ltron.dataset.brick_fill as brick_fill

def compute_logp(choice, choices, p=None):
    if p is None:
        return numpy.log2(1./len(choices))
    else:
        i = choices.index(choice)
        return numpy.log2(p[i])

def choice_logp(choices, p=None):
    choice = numpy.random.choice(choices, p=p)
    logp = compute_logp(choice, choices, p)
    return choice, logp

def sample_opaque_color():
    return choice_logp((
        0,  # black
        1,  # blue
        4,  # red
        5,  # pink
        7,  # light gray
        8,  # dark gray
        9,  # light blue
        10, # green
        11, # cyan
        12, # salmon
        14, # yellow
        15, # white
        22, # purple
        25, # orange
        29, # light pink
        31, # light purple
        70, # brown
        128,# light brown
        288,# dark green
        308,# dark brown
        320,# dark red
    ))

def sample_light_color():
    return choice_logp((
        36, # red
        38, # orange
        46, # yellow
        35, # green,
        33, # blue,
    ))

def add_doodads(scene, snaps):
    logp = 0.
    doodad_shape, doodad_shape_logp = choice_logp(
        ('6141.dat','3062.dat', '3070.dat', '4589.dat'),
        (0.25,0.25,0.25,0.25),
    )
    logp += doodad_shape_logp
    doodad_color, doodad_color_logp = sample_light_color()
    logp += doodad_color_logp
    
    for snap in snaps:
        instance = scene.add_instance(
            doodad_shape, doodad_color, scene.upright)
        
        doodad_snap = [
            snap for snap in instance.snaps
            if isinstance(snap.snap_style, StudHole)
        ][0]
        
        scene.pick_and_place_snap(doodad_snap, snap)
    
    return instance, logp

def sample_windshield_color():
    return choice_logp((40,43,47))

def sample_vehicle():
    scene = BrickScene()
    
    logp = 0.
    
    chassis_instances, dimensions, chassis_logp = sample_chassis(scene)
    logp += chassis_logp
    
    body_instances, dimensions, body_logp = sample_body(scene, dimensions)
    logp += body_logp
    
    '''
    roof_instances, dimensions, roof_logp = sample_roof(scene, dimensions)
    logp += roof_logp
    '''
    return scene, logp

# CHASSIS

def sample_chassis(scene):
    
    logp = 0.
    
    # pick a chassis type
    chassis_type, chassis_logp = choice_logp(
        ['two_wide', 'four_wide'], [1., 0.])
    logp += chassis_logp
    
    # separate axles connected to central 2-wide base
    if chassis_type == 'two_wide':
        chassis_instances, dimensions, chassis_logp =  sample_two_wide_chassis(
            scene)
    
    # separate axles connected to a central 4-wide base
    elif chassis_type == 'four_wide':
        chassis_instances, dimensions, chassis_logp = sample_four_wide_chassis(
            scene)
    
    # 2441.dat base, all-in-one
    # (see 10036 - Pizza To Go.ldr)
    elif chassis_type == '2441.dat':
        pass
    
    # 30029.dat base with no axles, but space for them
    # (see 10184 - Town Plan.mpd)
    elif chassis_type == '30029.dat':
        pass
    
    logp += chassis_logp
    
    return chassis_instances, dimensions, logp

def sample_two_wide_chassis(scene):
    logp = 0.
    
    # pick a thickness
    chassis_thickness, chassis_thickness_logp = choice_logp(
        ['plate', 'brick'], [0.5, 0.5])
    logp += chassis_thickness_logp
    
    # pick a chassis color
    chassis_color, chassis_color_logp = sample_opaque_color()
    logp += chassis_color_logp
    
    # pick split/single
    split_types = [False, True]
    split_p = [0.5, 0.5]
    chassis_split, chassis_split_logp = choice_logp(split_types, p=split_p)
    logp += chassis_split_logp
    
    # pick a chassis length
    if chassis_thickness == 'plate':
        chassis_length, chassis_length_logp = choice_logp([4,6,8,10,12])
        logp += chassis_length_logp
        body_height = 1
    else:
        if chassis_split:
            chassis_length, chassis_length_logp = choice_logp([4,6,8,10,12])
            logp += chassis_length_logp
        else:
            chassis_length, chassis_length_logp = choice_logp([4,6,8,10])
            logp += chassis_length_logp
        body_height = 3
    
    if chassis_split:
        left_instances = brick_fill.brick_fill(
            scene,
            -1, 0, 0, body_height, -chassis_length//2, chassis_length//2,
            chassis_color,
            start='min_xz',
        )
        right_instances = brick_fill.mirror(scene, left_instances, 0, 0)
        chassis_instances = left_instances + right_instances
    else:
        chassis_instances = brick_fill.brick_fill(
            scene,
            -1, 1, 0, body_height, -chassis_length//2, chassis_length//2,
            chassis_color,
            start='min_xz',
        )
    
    dimensions = -1, 1, -chassis_length//2, chassis_length//2, body_height
    
    # 2926.dat 4 wide single small axle
    # (see 10128-1 - Train Level Crossing.mpd)
    # 6157.dat 2+wide single small axles
    # (see 10156-1 - LEGO Truck.mpd)
    # 4600.dat 2 wide single small axles
    # (see 10159-1 - City Airport -City Logo Box.mpd)
    # 47720.dat 2 wide single axles, peg axles
    # (see 10197-1 - Fire Brigade.mpd)
    #if chassis_length > 4:
    #    axle_shape = numpy.random.choice(
    #        ['2926.dat', '6157.dat', '4600.dat', '47720.dat'],
    #        p=[0.25,0.25,0.25,0.25],
    #    )
    #else:
    
    # pick an axle type
    #['2926.dat', '6157.dat', '4600.dat'],
    axle_shape, axle_shape_logp = choice_logp(
       ['6157.dat', '4600.dat'], [0.5, 0.5])
    logp += axle_shape_logp
    
    # pick a rear axle position
    rear_axle_z = numpy.random.randint(-chassis_length//2, -1)
    logp += numpy.log2(1./len(range(-chassis_length//2, -1)))
    
    # pick a fore axle position
    fore_axle_z = numpy.random.randint(0, chassis_length//2-1)
    logp += numpy.log2(1./len(range(0, chassis_length//2-1)))
    
    # pick an axle color
    axle_chassis_color, axle_chassis_color_logp = choice_logp((True, False))
    logp += axle_chassis_color_logp
    if axle_chassis_color:
        axle_color = chassis_color
    else:
        axle_color, axle_color_logp = sample_opaque_color()
        logp += axle_color_logp
    
    # make the axles
    rear_axle_instance = make_axle(
        scene, axle_shape, axle_color, 0, 0, rear_axle_z)
    fore_axle_instance = make_axle(
        scene, axle_shape, axle_color, 0, 0, fore_axle_z)
    
    wheel_shape, wheel_color, tire_shape, tire_color, wheel_logp = (
        sample_wheel_for_axle(axle_shape, [rear_axle_z, fore_axle_z]))
    logp += wheel_logp
    
    rear_wheels, rear_tires = add_wheels_to_axle(
        scene,
        rear_axle_instance,
        wheel_shape,
        wheel_color,
        tire_shape,
        tire_color,
    )
    
    fore_wheels, fore_tires = add_wheels_to_axle(
        scene,
        fore_axle_instance,
        wheel_shape,
        wheel_color,
        tire_shape,
        tire_color,
    )
    
    # pick a fender color
    fender_chassis_color, fender_chassis_color_logp = choice_logp(
        (True, False), p=(0.5,0.5))
    logp += fender_chassis_color_logp
    if fender_chassis_color:
        fender_color = chassis_color
    else:
        fender_color, fender_color_logp = sample_opaque_color()
        logp += fender_color_logp
    
    # sample fender instances
    fender_instances, dimensions, fender_logp = sample_two_wide_fenders(
        scene,
        axle_shape,
        tire_shape,
        rear_tires+fore_tires,
        fender_color,
        (rear_axle_z, fore_axle_z),
        dimensions,
    )
    logp += fender_logp
    
    return None, dimensions, logp

def make_axle(scene, axle_shape, axle_color, x, y, z):
    translate = translate_matrix((x*20,y*8,z*20))
    if axle_shape == '2926.dat':
        translate[2,3] += 10
    if axle_shape == '6157.dat':
        translate[1,3] += 8
        translate[2,3] += 20
    if axle_shape == '4600.dat':
        translate[2,3] += 20
    if axle_shape == '47720.dat':
        translate[2,3] += 20
    
    instance = scene.add_instance(
        axle_shape, axle_color, translate @ scene.upright)
    
    return instance

def sample_two_wide_fenders(
    scene,
    axle_shape,
    tire_shape,
    tire_instances,
    color,
    zs,
    dimensions,
):
    logp = 0.
    x1,x2,z1,z2,y = dimensions
    
    fender_instances = []
    
    # build the chassis up higher if the tires are large
    if tire_shape in ('4084.dat', '6015.dat'):
        while y < tire_instances[0].transform[1,3] // 8 + 3:
            if y - tire_instances[0].transform[1,3] // 8 >= 3:
                step = 3
            else:
                step = 1
            brick_fill.brick_fill(
                scene,
                x1, x2, y, y+step,z1,z2,
                color,
                instances=list(scene.instances.values()),
            )
            y += step
    
    if (tire_shape in ('3641.dat', '4084.dat', '6015.dat') and
        axle_shape in ('6157.dat', '4600.dat')
    ):
        # if 6157, and y == 1 add another layer to the chassis
        if axle_shape == '6157.dat' and y == 1:
            brick_fill.brick_fill(
                scene,
                x1,x2,y,y+1,z1,z2,
                color,
                instances=list(scene.instances.values()),
            )
            y += 1
        
        # pick a fender shape
        fender_shape, fender_shape_logp = choice_logp(
            ('3788.dat', 'fill'), (0.75, 0.25))
        logp == fender_shape_logp
        z_offset = 1
        y_offset = 1
        
        if fender_shape == 'fill':
            # add another layer to the chassis
            brick_fill.brick_fill(
                scene,
                x1,x2,y,y+1,z1,z2,
                color,
                instances=list(scene.instances.values()),
            )
            y += 1
            
            # expand the fenders
            expand_fenders, expand_fenders_logp = choice_logp(
                (0,1), (0.25,0.75))
            logp += expand_fenders_logp
            if expand_fenders:
                x1 -= 1
                x2 += 1
            
            brick_fill.brick_fill(
                scene,
                x1, x2, y, y+1, z1, z2,
                color,
                instances=list(scene.instances.values()),
            )
            y += 1
            
        elif fender_shape == '3788.dat':
            # make the fender instances
            for z in zs:
                fender_translate = translate_matrix(
                    (0,(y+1+y_offset)*8,(z+z_offset)*20))
                fender_transform = fender_translate @ scene.upright
                fender_instance = scene.add_instance(
                    fender_shape,color, fender_transform)
                fender_instances.append(fender_instance)
            
            # fill the chassis between the fenders
            brick_fill.brick_fill(
                scene,
                x1, x2, y, y+1, z1, z2,
                color,
                #layer_mask,
                instances=list(scene.instances.values()),
            )
            y += 1
            
            # expand the fenders
            expand_fenders, expand_fenders_logp = choice_logp(
                (0,1), (0.25,0.75))
            logp += expand_fenders_logp
            if expand_fenders:
                x1 -= 1
                x2 += 1
            
            brick_fill.brick_fill(
                scene,
                x1, x2, y, y+1, z1, z2,
                color,
                instances=list(scene.instances.values()),
            )
            
            for z in zs:
                brick_fill.make_and_place(
                    scene,
                    -1, 1, y, y+1, z, z+2,
                    color,
                )
            
            y += 1
        
        dimensions = x1,x2,z1,z2,y
        
    else:
        fender_instances = []
    
    return fender_instances, dimensions, logp

def sample_wheel_for_axle(axle_shape, axle_zs):
    logp = 0.
    axle_spacing = min(z2-z1 for z1,z2 in zip(axle_zs[:-1], axle_zs[1:]))
    if axle_shape in ('2926.dat', '6157.dat', '4600.dat'):
        
        # pick a wheel shape
        if axle_spacing >= 3:
            wheel_shape, wheel_shape_logp = choice_logp(
                ('4624.dat','6014.dat'), (0.5,0.5))
            logp += wheel_shape_logp
        else:
            wheel_shape = '4624.dat'
        
        # pick a wheel color
        wheel_color, wheel_color_logp = choice_logp((15,), (1.,))
        logp += wheel_color_logp
        
        # pick a tire shape
        if axle_spacing >= 3:
            if wheel_shape == '4624.dat':
                tire_shape, tire_shape_logp = choice_logp(
                    ('3641.dat','4084.dat'), (0.5,0.5))
                logp += tire_shape_logp
            else:
                tire_shape = '6015.dat'
        else:
            tire_shape = '3641.dat'
        
        # pick a tire color
        tire_color, tire_color_logp = choice_logp((0,), (1.,))
        logp += tire_color_logp
        #min_body_height = {
        #    '3641.dat' : 2
        #}[tire_shape]
        return (
            wheel_shape,
            wheel_color,
            tire_shape,
            tire_color,
            logp,
            #min_body_height
        )

def add_wheels_to_axle(
    scene, axle_instance, wheel_shape, wheel_color, tire_shape, tire_color
):
    wheel_instances = []
    tire_instances = []
    used_snaps = set()
    for i in (2,1):
        wheel_instance = scene.add_instance(
            wheel_shape, wheel_color, numpy.eye(4))
        tire_instance = scene.add_instance(
            tire_shape, tire_color, numpy.eye(4))
        
        compatible_axle_snaps = [
            (j, snap) for (j, snap) in enumerate(axle_instance.snaps)
            if snap.compatible(wheel_instance.snaps[0])
            and j not in used_snaps
        ]
        
        assert len(compatible_axle_snaps) == i
        j, compatible_axle_snap = compatible_axle_snaps[0]
        used_snaps.add(j)
        
        scene.pick_and_place_snap(
            wheel_instance.snaps[0],
            compatible_axle_snap,
        )
        
        compatible_wheel_snaps = [
            (j, snap) for (j, snap) in enumerate(wheel_instance.snaps)
            if snap.compatible(tire_instance.snaps[0])
        ]
        assert len(compatible_wheel_snaps) == 1
        j, compatible_wheel_snap = compatible_wheel_snaps[0]
        
        scene.pick_and_place_snap(
            tire_instance.snaps[0],
            compatible_wheel_snap,
        )
        
        #scene.move_instance(tire_instance, wheel_instance.transform)
        
        wheel_instances.append(wheel_instance)
        tire_instances.append(tire_instance)
    
    return wheel_instances, tire_instances

# BODY

def sample_body(scene, dimensions):
    logp = 0.
    
    x1, x2, z1, z2, y = dimensions
    
    '''
    body_width, body_width_logp = choice_logp(
        (2,4), (0.2,0.8))
    logp += body_width_logp
    '''
    body_width = x2 - x1
    
    current_width = x2 - x1
    current_length = z2 - z1
    if current_width == 2 and body_width == 4:
        # expand out
        # 3665.dat (1x2) # messed up studs
        # 3660.dat (2x2) # messed up studs
        
        # 3747.dat (2x3)
        pass
    
    elif current_width == 4 and body_width == 2:
        # contract in
        # 3040.dat (1x2)
        # 3039.dat (2x2)
        # 3038.dat (3x2)
        
        # 4286.dat (1x3)
        # 3298.dat (2x3)
        # 3297.dat (4x3)
        pass
    
    body_color, color_logp = sample_opaque_color()
    logp += color_logp
    
    make_lower_wings, make_lower_wings_logp = choice_logp((0,1), (0.5,0.5))
    logp += make_lower_wings_logp
    if make_lower_wings:
        lower_wing_shapes, lower_wing_shapes_logp = choice_logp(
            ('50304.dat,50305.dat',
             '2450.dat,2450.dat'),
            (0.5, 0.5),
        )
        logp += lower_wing_shapes_logp
        
        wing_copy_body_color, wing_copy_body_color_logp = choice_logp(
            (0,1), (0.25, 0.75))
        logp += wing_copy_body_color_logp
        if wing_copy_body_color:
            wing_color = body_color
        else:
            wing_color, wing_color_logp = sample_opaque_color()
            logp += wing_color_logp
        
        wing_shape_a, wing_shape_b = lower_wing_shapes.split(',')
        #if lower_wing_shapes == '54383.dat,54384.dat':
        if lower_wing_shapes == '50304.dat,50305.dat':
            wing_z, wing_z_logp = choice_logp(list(range(z1+2, z2-1)))
            logp += wing_z_logp
            rotate_a = numpy.array([
                [ 0, 0, 1, 0],
                [ 0, 1, 0, 0],
                [-1, 0, 0, 0],
                [ 0, 0, 0, 1],
            ])
            if x2-x1 == 2:
                x_offset=3
            else:
                x_offset=2
            transform_a = rotate_a @ scene.upright
            transform_a[0,3] = (x1-x_offset) * 20
            transform_a[1,3] = (y+1) * 8
            transform_a[2,3] = (wing_z-0.5) * 20
            
            rotate_b = numpy.array([
                [ 0, 0,-1, 0],
                [ 0, 1, 0, 0],
                [ 1, 0, 0, 0],
                [ 0, 0, 0, 1],
            ])
            transform_b = rotate_b @ scene.upright
            transform_b[0,3] = (x2+x_offset)*20
            transform_b[1,3] = (y+1)*8
            transform_b[2,3] = (wing_z-0.5) * 20
            
            instance_a = scene.add_instance(
                wing_shape_a, wing_color, transform_a)
            instance_b = scene.add_instance(
                wing_shape_b, wing_color, transform_b)
            
            do_wing_doodad, do_wing_doodad_logp = choice_logp((0,1), (0.5,0.5))
            logp += do_wing_doodad_logp
            if do_wing_doodad:
                wing_doodad, wing_doodad_logp = add_doodads(
                    scene, [instance_a.snaps[2], instance_b.snaps[2]])
                logp += wing_doodad_logp
        
        elif lower_wing_shapes == '2450.dat,2450.dat':
            wing_z, wing_z_logp = choice_logp(list(range(z1+2, z2-1)))
            logp += wing_z_logp
            rotate_a = numpy.array([
                [ 1, 0, 0, 0],
                [ 0, 1, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1],
            ])
            transform_a = rotate_a @ scene.upright
            transform_a[0,3] = (x1-0.5) * 20
            transform_a[1,3] = (y+1) * 8
            transform_a[2,3] = (wing_z-0.5) * 20
            
            rotate_b = numpy.array([
                [ 0, 0,-1, 0],
                [ 0, 1, 0, 0],
                [ 1, 0, 0, 0],
                [ 0, 0, 0, 1],
            ])
            transform_b = rotate_b @ scene.upright
            transform_b[0,3] = (x2+0.5)*20
            transform_b[1,3] = (y+1)*8
            transform_b[2,3] = (wing_z-0.5) * 20
            
            instance_a = scene.add_instance(
                wing_shape_a, wing_color, transform_a)
            instance_b = scene.add_instance(
                wing_shape_b, wing_color, transform_b)
            
            do_wing_doodad, do_wing_doodad_logp = choice_logp((0,1), (0.5,0.5))
            logp += do_wing_doodad_logp
            if do_wing_doodad:
                wing_doodad, wing_doodad_logp = add_doodads(
                    scene, [instance_a.snaps[0], instance_b.snaps[5]])
                logp += wing_doodad_logp
                
        
        brick_fill.brick_fill(
            scene,
            x1, x2, y, y+1, z1, z2,
            body_color,
            instances=list(scene.instances.values()),
        )
        
        y += 1
    
    lower_height, lower_height_logp = choice_logp(
        (1,3,6,9), (0.1, 0.6, 0.2, 0.1))
    logp += lower_height_logp
    lower_y = y + lower_height
    
    while y < lower_y:
        if lower_y - y >= 3:
            y_step = 3
        else:
            y_step = 1
        brick_fill.brick_fill(
            scene,
            x1, x2, y, y+y_step, z1, z2,
            body_color,
        )
        y += y_step
    
    dimensions = x1, x2, z1, z2, y
    
    (shield_instances,
     shield_logp,
     rear_dimensions,
     roof_dimensions) = sample_windshield(
        scene, (x1, x2, z1, z2, y))
    
    roof_x1, roof_x2, roof_z1, roof_z2, roof_y = roof_dimensions
    rear_x1, rear_x2, rear_z1, rear_z2, rear_y = rear_dimensions
    
    while y < roof_y:
        if roof_y - y >=3:
            y_step = 3
        else:
            y_step = 1
        brick_fill.brick_fill(
            scene,
            rear_x1, rear_x2, y, y+y_step, rear_z1, rear_z2,
            body_color,
        )
        y += y_step 
    
    roof_body_color, roof_body_color_logp = choice_logp(
        (True, False), (0.5, 0.5))
    logp += roof_body_color_logp
    if roof_body_color:
        roof_color = body_color
    else:
        roof_color, roof_color_logp = sample_opaque_color()
        logp += roof_color_logp
    brick_fill.brick_fill(
        scene,
        roof_x1, roof_x2, y, y+1, roof_z1, roof_z2,
        roof_color,
    )
    
    return [], dimensions, logp

def sample_windshield(scene, dimensions):
    logp = 0.
    
    x1, x2, z1, z2, y = dimensions
    width = x2 - x1
    
    instances = []
    
    if width == 4:
        windshield_shape, windshield_shape_logp = choice_logp(
            ('3823.dat','4594.dat','2437.dat', '4866.dat', '4215.dat'),
            (0.2,0.2,0.2,0.2,0.2),
        )
        logp += windshield_shape_logp
        windshield_color, windshield_color_logp = sample_windshield_color()
        logp += windshield_color_logp
        windshield_transform = scene.upright.copy()
        if windshield_shape == '3823.dat':
            windshield_transform[1,3] = (y+6)*8
            windshield_transform[2,3] = (z1+1.5)*20
            roof_y = y+6
            roof_z1 = z1+1
            rear_z1 = z1+2
        elif windshield_shape == '4594.dat':
            windshield_transform[1,3] = (y+6)*8
            windshield_transform[2,3] = (z1+1.5)*20
            roof_y = y+6
            roof_z1 = z1
            rear_z1 = z1+2
        elif windshield_shape == '2437.dat':
            windshield_transform[1,3] = (y+4)*8
            windshield_transform[2,3] = (z1+1.5)*20
            roof_y = y+4
            roof_z1 = z1+1
            rear_z1 = z1+3
        elif windshield_shape == '4866.dat':
            windshield_transform[1,3] = (y+4)*8
            windshield_transform[2,3] = (z1+1.5)*20
            roof_y = y+4
            roof_z1 = z1+1
            rear_z1 = z1+3
        elif windshield_shape == '4215.dat':
            windshield_transform[0,0] *= -1
            windshield_transform[2,2] *= -1
            windshield_transform[1,3] = (y+9)*8
            windshield_transform[2,3] = (z1+0.5)*20
            roof_y = y+9
            roof_z1 = z1
            rear_z1 = z1+3
        
        i = scene.add_instance(
            windshield_shape, windshield_color, windshield_transform)
        instances.append(i)
    
    elif width == 2:
        windshield_shape, windshield_shape_logp = choice_logp(
            ('2466.dat', '4864.dat'), (0.,1.)
        )
        logp += windshield_shape_logp
        windshield_color, windshield_color_logp = sample_windshield_color()
        logp += windshield_color_logp
        windshield_transform = scene.upright.copy()
        if windshield_shape == '2466.dat':
            windshield_transform[1,3] = (y+17)*8
            windshield_transform[2,3] = (z1+0.5)*20
            roof_y = y + 17
            roof_z1 = z1
            rear_z1 = z1+3
        
        if windshield_shape == '4864.dat':
            windshield_transform[0,0] *= -1
            windshield_transform[2,2] *= -1
            windshield_transform[1,3] = (y+6)*8
            windshield_transform[2,3] = (z1+0.5)*20
            roof_y = y + 6
            roof_z1 = z1
            rear_z1 = z1+3
        
        i = scene.add_instance(
            windshield_shape, windshield_color, windshield_transform)
        instances.append(i)
    
    rear_dimensions = (x1,x2,rear_z1,z2,y)
    roof_dimensions = (x1,x2,roof_z1,z2,roof_y)
    return instances, logp, rear_dimensions, roof_dimensions

if __name__ == '__main__':
    scene, logp = sample_vehicle()
    print(logp)
    scene.export_ldraw('./tmp.mpd')
