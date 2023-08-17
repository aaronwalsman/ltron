import numpy

from ltron.bricks.brick_scene import BrickScene
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
    return choice_logp((0,1,2,4,5,6,7,8,10,12,14))

def sample_vehicle():
    scene = BrickScene()
    
    logp = 0.
    
    chassis_instances, dimensions, chassis_logp = sample_chassis(scene)
    logp += chassis_logp
    
    '''
    body_instances, dimensions, body_logp = sample_body(scene, dimensions)
    logp += body_logp
    
    roof_instances, dimensions, roof_logp = sample_roof(scene, dimensions)
    logp += roof_logp
    '''
    return scene, logp
    
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
        sample_wheel_for_axle(axle_shape))
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
    fender_instances, dimensions = sample_two_wide_fenders(
        scene,
        axle_shape,
        tire_shape,
        fender_color,
        #body_height,
        (rear_axle_z, fore_axle_z),
        dimensions,
        #(2,chassis_length)
    )
    
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
    color,
    #y,
    zs,
    #chassis_shape,
    dimensions,
):
    x1,x2,z1,z2,y = dimensions
    
    if tire_shape == '3641.dat' and axle_shape in ('6157.dat', '4600.dat'):
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
        fender_shape = '3788.dat'
        z_offset = 1
        y_offset = 1
        fender_height = 2
        
        # make the fender instances
        fender_instances = []
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
        expand_fenders = 1
        if expand_fenders:
            brick_fill.brick_fill(
                scene,
                x1-1, x2+1, y, y+1, z1, z2,
                color,
                instances=list(scene.instances.values()),
            )
            
            for z in zs:
                brick_fill.make_and_place(scene, -1, 1, y, y+1, z, z+2, color)
            
            y += 1
        
    else:
        fender_instances = []
        fender_height = 0
    return fender_instances, fender_height

def sample_wheel_for_axle(axle_shape):
    logp = 0.
    if axle_shape in ('2926.dat', '6157.dat', '4600.dat'):
        
        # pick a wheel shape
        wheel_shape, wheel_shape_logp = choice_logp(
            ('4624.dat',), (1.0,))
        logp += wheel_shape_logp
        
        # pick a wheel color
        wheel_color, wheel_color_logp = choice_logp((15,), (1.,))
        logp += wheel_color_logp
        
        # pick a tire shape
        tire_shape, tire_shape_logp = choice_logp(
            ('3641.dat',), (1.,))
        logp += tire_shape_logp
        
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
            (i, snap) for (i, snap) in enumerate(axle_instance.snaps)
            if snap.compatible(wheel_instance.snaps[0])
            and i not in used_snaps
        ]
        
        if not len(compatible_axle_snaps) == i:
            breakpoint()
        assert len(compatible_axle_snaps) == i
        i, compatible_axle_snap = compatible_axle_snaps[0]
        used_snaps.add(i)
        
        scene.pick_and_place_snap(
            wheel_instance.snaps[0],
            compatible_axle_snap,
        )
        
        scene.move_instance(tire_instance, wheel_instance.transform)
        
        wheel_instances.append(wheel_instance)
        tire_instances.append(tire_instance)
    
    return wheel_instances, tire_instances

if __name__ == '__main__':
    scene, logp = sample_vehicle()
    print(logp)
    scene.export_ldraw('./tmp.mpd')
