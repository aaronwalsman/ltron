import os
import sys

import numpy

import tqdm

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

def sample_wheel_color():
    return choice_logp((0,7,15))

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

def sample_opaque_color_palette():
    num_colors, num_colors_logp = choice_logp(
        [1,2,3,4,5], [0.05,0.35,0.35,0.2,0.05])
    colors, color_logps = zip(
        *[sample_opaque_color() for _ in range(num_colors)])
    return colors, num_colors_logp + sum(color_logps)

def sample_light_color():
    return choice_logp((
        36, # red
        38, # orange
        46, # yellow
        35, # green,
        33, # blue,
    ))

def add_tall_doodads(scene, snaps):
    color_logp = 0.
    shape_logp = 0.
    doodad_shape, doodad_shape_logp = choice_logp(
        ('4589.dat', '3062.dat'),
        (0.5,0.5),
    )
    shape_logp += doodad_shape_logp
    doodad_color, doodad_color_logp = sample_light_color()
    color_logp += doodad_color_logp
    
    instances = []
    for snap in snaps:
        instances.append(scene.add_instance(
            doodad_shape, doodad_color, scene.upright))
        
        doodad_snap = [
            snap for snap in instances[-1].snaps
            if isinstance(snap.snap_style, StudHole)
        ][0]
        
        scene.pick_and_place_snap(doodad_snap, snap)
    
    return instances, shape_logp, color_logp

def add_small_doodads(scene, snaps):
    color_logp = 0.
    shape_logp = 0.
    doodad_shape, doodad_shape_logp = choice_logp(
        ('6141.dat', '3070.dat', '3024.dat', '98138.dat'),
        (0.25,0.25,0.25,0.25),
    )
    shape_logp += doodad_shape_logp
    doodad_color, doodad_color_logp = sample_light_color()
    color_logp += doodad_color_logp
    
    instances = []
    for snap in snaps:
        instances.append(scene.add_instance(
            doodad_shape, doodad_color, scene.upright))
        
        doodad_snap = [
            snap for snap in instances[-1].snaps
            if isinstance(snap.snap_style, StudHole)
        ][0]
        
        scene.pick_and_place_snap(doodad_snap, snap)
    
    return instances, shape_logp, color_logp

def add_stickups(
    scene,
    color_palette,
    snaps=None,
    locations=None,
):
    color_logp = 0.
    shape_logp = 0.
    
    stickup_color, stickup_color_logp = choice_logp(color_palette)
    color_logp += stickup_color_logp
    
    instances = []
    doodad_snaps = []
    if snaps is not None:
        for snap in snaps:
            #stickup_transform = numpy.array([
            #    [ 0, 0, 1, 0],
            #    [ 0, 1, 0, 0],
            #    [-1, 0, 0, 0],
            #    [ 0, 0, 0, 1],
            #])@scene.upright
            stickup_instance = scene.add_instance(
                '4070.dat', stickup_color, scene.upright)
            instances.append(stickup_instance)
            scene.pick_and_place_snap(stickup_instance.snaps[2], snap)
            doodad_snaps.append(stickup_instance.snaps[1])
    
    elif locations is not None:
        for location in locations:
            transform = numpy.array([
                [1, 0, 0, location[0]*20-10],
                [0, 1, 0, location[1]*8+16],
                [0, 0, 1, location[2]*20-10],
                [0, 0, 0, 1],
            ]) @ scene.upright
            stickup_instance = scene.add_instance(
                '4070.dat', stickup_color, transform)
            instances.append(stickup_instance)
            doodad_snaps.append(stickup_instance.snaps[1])
    
    doodad_tall, doodad_tall_logp = choice_logp((0,1))
    shape_logp += doodad_tall_logp

    if doodad_tall:
        (doodad_instances,
         doodad_color_logp,
         doodad_shape_logp) = add_tall_doodads(scene, doodad_snaps)
        color_logp += doodad_color_logp
        shape_logp += doodad_shape_logp
    else:
        (doodad_instances,
         doodad_color_logp,
         doodad_shape_logp) = add_small_doodads(scene, doodad_snaps)
        color_logp += doodad_color_logp
        shape_logp ++ doodad_shape_logp
    
    return instances, shape_logp, color_logp

def add_stickouts(
    scene,
    color_palette,
    right_snap=None,
    left_snap=None,
    right_location=None,
    left_location=None,
):
    color_logp = 0.
    shape_logp = 0.
    
    stickout_color, stickout_color_logp = choice_logp(color_palette)
    color_logp += stickout_color_logp
    
    right_stickout_transform = numpy.array([
        [ 0, 0,-1, 0],
        [ 0, 1, 0, 0],
        [ 1, 0, 0, 0],
        [ 0, 0, 0, 1],
    ]) @ scene.upright
    if right_location is not None:
        right_stickout_transform[:3,3] = [
            right_location[0]*20-10,
            right_location[1]*8,
            right_location[2]*20-10,
        ]
    right_instance = scene.add_instance(
        '4081.dat', stickout_color, right_stickout_transform)
    
    if right_snap is not None:
        scene.pick_and_place_snap(right_instance.snaps[4], right_snap)
    
    left_stickout_transform = numpy.array([
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [-1, 0, 0, 0],
        [ 0, 0, 0, 1],
    ]) @ scene.upright
    if left_location is not None:
        left_stickout_transform[:3,3] = [
            left_location[0]*20+10,
            left_location[1]*8,
            left_location[2]*20-10,
        ]
    left_instance = scene.add_instance(
        '4081.dat', stickout_color, left_stickout_transform)
    
    if left_snap is not None:
        scene.pick_and_place_snap(left_instance.snaps[4], left_snap)
    
    doodad_tall, doodad_tall_logp = choice_logp((0,1))
    shape_logp += doodad_tall_logp
    
    if doodad_tall:
        (doodad_instances,
         doodad_color_logp,
         doodad_shape_logp) = add_tall_doodads(
            scene, [right_instance.snaps[1], left_instance.snaps[2]])
        color_logp += doodad_color_logp
        shape_logp += doodad_shape_logp
    else:
        (doodad_instances,
         doodad_color_logp,
         doodad_shape_logp) = add_small_doodads(
            scene, [right_instance.snaps[1], left_instance.snaps[2]])
        color_logp += doodad_color_logp
        shape_logp ++ doodad_shape_logp
    
    return (
        [right_instance, left_instance] + doodad_instances,
        color_logp,
        shape_logp,
    )

def sample_windshield_color():
    return choice_logp((43,47))

def sample_vehicle():
    scene = BrickScene()
    
    color_logp = 0.
    shape_logp = 0.
    
    color_palette, color_palette_logp = sample_opaque_color_palette()
    color_logp += color_palette_logp
    
    (chassis_instances,
     dimensions,
     chassis_color_logp,
     chassis_shape_logp) = sample_chassis(scene, color_palette)
    color_logp += chassis_color_logp
    shape_logp += chassis_shape_logp
    
    (body_instances,
     dimensions,
     body_color_logp,
     body_shape_logp) = sample_body(scene, dimensions, color_palette)
    color_logp += body_color_logp
    shape_logp += body_shape_logp
    
    (roof_instances,
     dimensions,
     roof_color_logp,
     roof_shape_logp) = sample_roof(scene, dimensions, color_palette)
    color_logp += roof_color_logp
    shape_logp += roof_shape_logp
    
    return scene, color_logp, shape_logp

# CHASSIS

def sample_chassis(scene, color_palette):
    
    color_logp = 0.
    shape_logp = 0.
    
    # pick a chassis type
    chassis_type, chassis_logp = choice_logp(
        ['two_wide', 'four_wide', 'helicopter'], [1., 0., 0.])
    shape_logp += chassis_logp
    
    # separate axles connected to central 2-wide base
    if chassis_type == 'two_wide':
        (chassis_instances,
         dimensions,
         chassis_color_logp,
         chassis_shape_logp) = sample_two_wide_chassis(scene, color_palette)
        color_logp += chassis_color_logp
        shape_logp += chassis_shape_logp
    
    # separate axles connected to a central 4-wide base
    elif chassis_type == 'four_wide':
        (chassis_instances,
         dimensions,
         chassis_color_logp,
         chassis_shape_logp) = sample_four_wide_chassis(scene, color_palette)
        color_logp += chassis_color_logp
        shape_logp += chassis_shape_logp
    
    elif chassis_type == 'helicopter':
        (chassis_instances,
         dimensions,
         chassis_color_logp,
         chassis_shape_logp) = sample_helicopter_chassis(scene, color_palette)
        color_logp += chassis_color_logp
        shape_logp += chassis_shape_logp
    
    # 2441.dat base, all-in-one
    # (see 10036 - Pizza To Go.ldr)
    elif chassis_type == '2441.dat':
        pass
    
    # 30029.dat base with no axles, but space for them
    # (see 10184 - Town Plan.mpd)
    elif chassis_type == '30029.dat':
        pass
    
    return chassis_instances, dimensions, color_logp, shape_logp

def sample_two_wide_chassis(scene, color_palette):
    color_logp = 0.
    shape_logp = 0.
    
    # pick a thickness
    #chassis_thickness, chassis_thickness_logp = choice_logp(
    #    ['plate', 'brick'], [0.5, 0.5])
    #shape_logp += chassis_thickness_logp
    chassis_thickness = 'plate'
    
    # pick a chassis color
    chassis_color, chassis_color_logp = choice_logp(color_palette)
    color_logp += chassis_color_logp
    
    # pick split/single
    #split_types = [False, True]
    #split_p = [0.5, 0.5]
    #chassis_split, chassis_split_logp = choice_logp(split_types, p=split_p)
    #shape_logp += chassis_split_logp
    chassis_split = False
    
    # pick a chassis length
    if chassis_thickness == 'plate':
        chassis_length, chassis_length_logp = choice_logp([4,6,8,10,12])
        shape_logp += chassis_length_logp
        body_height = 1
    else:
        if chassis_split:
            chassis_length, chassis_length_logp = choice_logp([4,6,8,10,12])
            shape_logp += chassis_length_logp
        else:
            chassis_length, chassis_length_logp = choice_logp([4,6,8,10])
            shape_logp += chassis_length_logp
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
       ['6157.dat', '4600.dat'], [0., 1.])
    shape_logp += axle_shape_logp
    
    # pick a rear axle position
    rear_axle_z = numpy.random.randint(-chassis_length//2, -1)
    shape_logp += numpy.log2(1./len(range(-chassis_length//2, -1)))
    
    # pick a fore axle position
    fore_axle_z = numpy.random.randint(0, chassis_length//2-1)
    shape_logp += numpy.log2(1./len(range(0, chassis_length//2-1)))
    
    # pick an axle color
    #axle_chassis_color, axle_chassis_color_logp = choice_logp((True, False))
    #color_logp += axle_chassis_color_logp
    #if axle_chassis_color:
    #    axle_color = chassis_color
    #else:
    axle_color, axle_color_logp = choice_logp(color_palette)
    color_logp += axle_color_logp
    
    # make the axles
    rear_axle_instance = make_axle(
        scene, axle_shape, axle_color, 0, 2, rear_axle_z)
    fore_axle_instance = make_axle(
        scene, axle_shape, axle_color, 0, 2, fore_axle_z)
    
    # make axle fill
    axle_fill_instances = brick_fill.brick_fill(
        scene,
        -1, 1, 1, 2, -chassis_length//2, chassis_length//2,
        chassis_color,
        start='min_xz',
        instances=list(scene.instances.values()),
    )
    
    # make plate above axles
    axle_top_instances = brick_fill.brick_fill(
        scene,
        -1, 1, 2, 3, -chassis_length//2, chassis_length//2,
        chassis_color,
        start='min_xz',
        instances=list(scene.instances.values()),
    )
    dimensions = -1, 1, -chassis_length//2, chassis_length//2, body_height+2
    
    (wheel_shape,
     wheel_color,
     tire_shape,
     tire_color,
     wheel_color_logp,
     wheel_shape_logp) = sample_wheel_for_axle(
        axle_shape, [rear_axle_z, fore_axle_z])
    color_logp += wheel_color_logp
    shape_logp += wheel_shape_logp
    
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
    #fender_chassis_color, fender_chassis_color_logp = choice_logp(
    #    (True, False), p=(0.5,0.5))
    #color_logp += fender_chassis_color_logp
    #if fender_chassis_color:
    #    fender_color = chassis_color
    #else:
    fender_color, fender_color_logp = choice_logp(color_palette)
    color_logp += fender_color_logp
    
    # sample fender instances
    (fender_instances,
     dimensions,
     fender_color_logp,
     fender_shape_logp) = sample_two_wide_fenders(
        scene,
        axle_shape,
        tire_shape,
        rear_tires+fore_tires,
        fender_color,
        (rear_axle_z, fore_axle_z),
        dimensions,
    )
    color_logp += fender_color_logp
    shape_logp += fender_shape_logp
    
    return None, dimensions, color_logp, shape_logp

def sample_helicopter_chassis(scene, color_palette):
    color_logp = 0.
    shape_logp = 0.
    
    # pick a thickness
    #chassis_thickness, chassis_thickness_logp = choice_logp(
    #    ['plate', 'brick'], [0.5, 0.5])
    #shape_logp += chassis_thickness_logp
    chassis_thickness = 'plate'
    
    # pick a chassis color
    chassis_color, chassis_color_logp = choice_logp(color_palette)
    color_logp += chassis_color_logp
    
    # pick split/single
    #split_types = [False, True]
    #split_p = [0.5, 0.5]
    #chassis_split, chassis_split_logp = choice_logp(split_types, p=split_p)
    #shape_logp += chassis_split_logp
    chassis_split = False
    
    # pick a chassis length
    if chassis_thickness == 'plate':
        chassis_length, chassis_length_logp = choice_logp([4,6,8,10,12])
        shape_logp += chassis_length_logp
        body_height = 1
    else:
        if chassis_split:
            chassis_length, chassis_length_logp = choice_logp([4,6,8,10,12])
            shape_logp += chassis_length_logp
        else:
            chassis_length, chassis_length_logp = choice_logp([4,6,8,10])
            shape_logp += chassis_length_logp
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
    
    body_width, body_width_logp = choice_logp([2,4], [0.2,0.8])
    shape_logp += body_width_logp
    if body_width == 4:
        xs = -2,2
        brick_fill.brick_fill(
            scene,
            -2,
            2,
            body_height,
            body_height+1,
            -chassis_length//2,
            chassis_length//2,
            chassis_color,
        )
        body_height += 1
    else:
        xs = -1,1
    
    dimensions = (*xs, -chassis_length//2, chassis_length//2, body_height)
    
    strut_color, strut_color_logp = choice_logp(color_palette)
    color_logp += strut_color_logp
    
    strut1_z, strut1_z_logp = choice_logp(
        range(-chassis_length//2, chassis_length//2-2))
    shape_logp += strut1_z_logp
    strut1_transform = numpy.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,strut1_z*20+10],
        [0,0,0,1],
    ]) @ scene.upright
    instance = scene.add_instance('6140.dat', strut_color, strut1_transform)
    
    strut2_z, strut2_z_logp = choice_logp(
        range(strut1_z+2, chassis_length//2))
    shape_logp += strut2_z_logp
    strut2_transform = numpy.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,strut2_z*20+10],
        [0,0,0,1],
    ]) @ scene.upright
    instance = scene.add_instance('6140.dat', strut_color, strut2_transform)
    
    ski_color, ski_color_logp = choice_logp(color_palette)
    color_logp += ski_color_logp
    
    strut_spacing = strut2_z - strut1_z
    ski_lengths = [
        ski_length for ski_length in [4,6,8,10,12]
        if strut_spacing < ski_length
    ]
    ski_length, ski_length_logp = choice_logp(ski_lengths)
    shape_logp += ski_length_logp
    
    extra_length = ski_length - strut_spacing
    ski_offset, ski_offset_logp = choice_logp(range(extra_length))
    shape_logp ++ ski_offset_logp
    
    ski1 = brick_fill.make_and_place(
        scene, -3, -2, -4, -3,
        strut1_z - ski_offset,
        strut1_z - ski_offset + ski_length,
        ski_color,
    )
    
    ski2 = brick_fill.make_and_place(
        scene, 2, 3, -4, -3,
        strut1_z - ski_offset,
        strut1_z - ski_offset + ski_length,
        ski_color,
    )
    
    return None, dimensions, color_logp, shape_logp

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
    color_logp = 0.
    shape_logp = 0.
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
        shape_logp == fender_shape_logp
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
            shape_logp += expand_fenders_logp
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
            shape_logp += expand_fenders_logp
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
    
    return fender_instances, dimensions, color_logp, shape_logp

def sample_wheel_for_axle(axle_shape, axle_zs):
    color_logp = 0.
    shape_logp = 0.
    
    axle_spacing = min(z2-z1 for z1,z2 in zip(axle_zs[:-1], axle_zs[1:]))
    if axle_shape in ('2926.dat', '6157.dat', '4600.dat'):
        
        # pick a wheel shape
        if axle_spacing >= 3:
            wheel_shape, wheel_shape_logp = choice_logp(
                ('4624.dat','6014.dat'), (0.5,0.5))
            shape_logp += wheel_shape_logp
        else:
            wheel_shape = '4624.dat'
        
        # pick a wheel color
        #wheel_color, wheel_color_logp = choice_logp((15,), (1.,))
        wheel_color, wheel_color_logp = sample_wheel_color()
        color_logp += wheel_color_logp
        
        # pick a tire shape
        if axle_spacing >= 3:
            if wheel_shape == '4624.dat':
                tire_shape, tire_shape_logp = choice_logp(
                    ('3641.dat','4084.dat'), (0.5,0.5))
                shape_logp += tire_shape_logp
            else:
                tire_shape = '6015.dat'
        else:
            tire_shape = '3641.dat'
        
        # pick a tire color
        tire_color, tire_color_logp = choice_logp((0,), (1.,))
        color_logp += tire_color_logp
        #min_body_height = {
        #    '3641.dat' : 2
        #}[tire_shape]
        return (
            wheel_shape,
            wheel_color,
            tire_shape,
            tire_color,
            color_logp,
            shape_logp,
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

def sample_body(scene, dimensions, color_palette):
    color_logp = 0.
    shape_logp = 0.
    
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
    
    body_color, color_logp = choice_logp(color_palette)
    color_logp += color_logp
    
    make_lower_wings, make_lower_wings_logp = choice_logp((0,1), (0.6,0.4))
    shape_logp += make_lower_wings_logp
    if make_lower_wings:
        if body_width == 4:
            lower_wing_shapes, lower_wing_shapes_logp = choice_logp(
                ('50304.dat,50305.dat',
                 '2450.dat,2450.dat',
                 '3475.dat,3475.dat'),
            )
            shape_logp += lower_wing_shapes_logp
        else:
            lower_wing_shapes, lower_wing_shapes_logp = choice_logp(
                ('50304.dat,50305.dat',
                 '2450.dat,2450.dat'),
            )
            shape_logp += lower_wing_shapes_logp
        
        wing_copy_body_color, wing_copy_body_color_logp = choice_logp(
            (0,1), (0.25, 0.75))
        color_logp += wing_copy_body_color_logp
        if wing_copy_body_color:
            wing_color = body_color
        else:
            wing_color, wing_color_logp = choice_logp(color_palette)
            color_logp += wing_color_logp
        
        wing_shape_a, wing_shape_b = lower_wing_shapes.split(',')
        #if lower_wing_shapes == '54383.dat,54384.dat':
        if lower_wing_shapes == '50304.dat,50305.dat':
            wing_z, wing_z_logp = choice_logp(list(range(z1+2, z2-1)))
            shape_logp += wing_z_logp
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
            
            #do_wing_doodad, do_wing_doodad_logp = choice_logp((0,1), (0.5,0.5))
            #logp += do_wing_doodad_logp
            wing_ornament, wing_ornament_logp = choice_logp(
                ('none', 'doodad', 'stickout', 'stickup'))
            shape_logp += wing_ornament_logp
            if wing_ornament == 'doodad':
                (wing_doodads,
                 wing_doodad_color_logp,
                 wing_doodad_shape_logp) = add_small_doodads(
                    scene, [instance_a.snaps[2], instance_b.snaps[2]])
                color_logp += wing_doodad_color_logp
                shape_logp += wing_doodad_shape_logp
            elif wing_ornament == 'stickout':
                (wing_stickouts,
                 wing_stickouts_color_logp,
                 wing_stickouts_shape_logp) = add_stickouts(
                    scene,
                    color_palette,
                    right_snap=instance_b.snaps[2],
                    left_snap=instance_a.snaps[2],
                )
                color_logp += wing_stickouts_color_logp
                shape_logp += wing_stickouts_shape_logp
            
            elif wing_ornament == 'stickup':
                (wing_stickups,
                 wing_stickups_color_logp,
                 wing_stickups_shape_logp) = add_stickups(
                    scene,
                    color_palette,
                    snaps=[instance_b.snaps[2], instance_a.snaps[2]],
                )
                color_logp += wing_stickups_color_logp
                shape_logp += wing_stickups_shape_logp
        
        elif lower_wing_shapes == '2450.dat,2450.dat':
            wing_z, wing_z_logp = choice_logp(list(range(z1+2, z2-1)))
            shape_logp += wing_z_logp
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
            
            #do_wing_doodad, do_wing_doodad_logp = choice_logp((0,1), (0.5,0.5))
            #logp += do_wing_doodad_logp
            #if do_wing_doodad:
            #    wing_doodads, wing_doodad_logp = add_small_doodads(
            #        scene, [instance_a.snaps[0], instance_b.snaps[5]])
            #    logp += wing_doodad_logp
            wing_ornament, wing_ornament_logp = choice_logp(
                ('none', 'doodad', 'stickout', 'stickup'))
            shape_logp += wing_ornament_logp
            if wing_ornament == 'doodad':
                (wing_doodads,
                 wing_doodad_color_logp,
                 wing_doodad_shape_logp) = add_small_doodads(
                    scene, [instance_a.snaps[0], instance_b.snaps[5]])
                color_logp += wing_doodad_color_logp
                shape_logp += wing_doodad_shape_logp
            elif wing_ornament == 'stickout':
                (wing_stickouts,
                 wing_stickouts_color_logp,
                 wing_stickouts_shape_logp) = add_stickouts(
                    scene,
                    color_palette,
                    right_snap=instance_b.snaps[5],
                    left_snap=instance_a.snaps[0],
                )
                color_logp += wing_stickouts_color_logp
                shape_logp += wing_stickouts_shape_logp
            elif wing_ornament == 'stickup':
                (wing_stickups,
                 wing_stickups_color_logp,
                 wing_stickups_shape_logp) = add_stickups(
                    scene,
                    color_palette,
                    snaps=[instance_b.snaps[5], instance_a.snaps[0]],
                )
        
        elif lower_wing_shapes == '3475.dat,3475.dat':
            #wing_z, wing_z_logp = choice_logp(list(range(z1+1, z2)))
            #logp += wing_z_logp
            wing_z = z2-1
            rotate_a = numpy.array([
                [ 0, 0,-1, 0],
                [ 0, 1, 0, 0],
                [ 1, 0, 0, 0],
                [ 0, 0, 0, 1],
            ])
            transform_a = rotate_a @ scene.upright
            transform_a[0,3] = (x1+0.5) * 20
            transform_a[1,3] = (y+1) * 8
            transform_a[2,3] = (wing_z) * 20
            
            rotate_b = numpy.array([
                [ 0, 0, 1, 0],
                [ 0, 1, 0, 0],
                [-1, 0, 0, 0],
                [ 0, 0, 0, 1],
            ])
            transform_b = rotate_b @ scene.upright
            transform_b[0,3] = (x2-0.5)*20
            transform_b[1,3] = (y+1)*8
            transform_b[2,3] = (wing_z) * 20
            
            instance_a = scene.add_instance(
                wing_shape_a, wing_color, transform_a)
            instance_b = scene.add_instance(
                wing_shape_b, wing_color, transform_b)
            
            if wing_z != z2-1:
                bridge_instance_a = brick_fill.make_and_place(
                    scene, x1, x2, y, y+1, wing_z+1, wing_z+2, body_color)
            if wing_z != z1+1:
                bridge_instance_b = brick_fill.make_and_place(
                    scene, x1, x2, y, y+1, wing_z-2, wing_z-1, body_color)
        
        brick_fill.brick_fill(
            scene,
            x1, x2, y, y+1, z1, z2,
            body_color,
            instances=list(scene.instances.values()),
        )
        
        y += 1
    
    make_bumper, make_bumper_logp = choice_logp((0,1), (0.5,0.5))
    shape_logp += make_bumper_logp
    if make_bumper:
        bumper_height, bumper_height_logp = choice_logp((1,3), (0., 1.))
        shape_logp += bumper_height_logp
        
        #bumper_color, bumper_color_logp = choice_logp(color_palette)
        bumper_color = body_color
        
        width = x2 - x1
        if bumper_height == 1:
            pass
        
        elif bumper_height == 3:
            if width == 2:
                bumper_transform = scene.upright.copy()
                bumper_transform[0,3] = 10
                bumper_transform[1,3] = (y+3) * 8
                bumper_transform[2,3] = z1*20 + 10
                bumper_a = scene.add_instance(
                    '4070.dat', bumper_color, bumper_transform)
                snap_a = bumper_a.snaps[1]
                
                bumper_transform = scene.upright.copy()
                bumper_transform[0,3] = -10
                bumper_transform[1,3] = (y+3) * 8
                bumper_transform[2,3] = z1*20 + 10
                bumper_b = scene.add_instance(
                    '4070.dat', bumper_color, bumper_transform)
                snap_b = bumper_b.snaps[1]
                
                do_grill, do_grill_logp = choice_logp((0,1))
                shape_logp += do_grill_logp
                if do_grill:
                    grill_transform = numpy.array([
                        [1, 0, 0, 0],
                        [0, 0, 1, (y+1)*8+6],
                        [0,-1, 0, z1*20 -4],
                        [0, 0, 0, 1]
                    ]) @ scene.upright.copy()
                    
                    grill_color, grill_color_logp = sample_wheel_color()
                    color_logp += grill_color_logp
                    instance = scene.add_instance(
                        '2412.dat', grill_color, grill_transform)
                else:
                    (doodads,
                     doodad_color_logp,
                     doodad_shape_logp) = add_small_doodads(
                        scene, [snap_a, snap_b])
                    doodad_color_logp += doodad_color_logp
                    doodad_shape_logp += doodad_shape_logp
            
            elif width == 4:
                bumper_transform = scene.upright.copy()
                bumper_transform[1,3] = (y+3) * 8
                bumper_transform[2,3] = z1*20 + 10
                bumper_brick = scene.add_instance(
                    '30414.dat', bumper_color, bumper_transform)
                
                grills, grills_logp = choice_logp((0,1,2))
                shape_logp += grills_logp
                if grills == 0:
                    (doodads,
                     doodad_color_logp,
                     doodad_shape_logp) = add_small_doodads(
                        scene, [bumper_brick.snaps[0], bumper_brick.snaps[3]])
                    color_logp += doodad_color_logp
                    shape_logp += doodad_shape_logp
                    (doodads,
                     doodad_color_logp,
                     doodad_shape_logp) = add_small_doodads(
                        scene, [bumper_brick.snaps[1], bumper_brick.snaps[2]])
                    color_logp += doodad_color_logp
                    shape_logp += doodad_shape_logp
                
                elif grills == 1:
                    grill_transform = numpy.array([
                        [1, 0, 0, 0],
                        [0, 0, 1, (y+1)*8+6],
                        [0,-1, 0, z1*20 -8],
                        [0, 0, 0, 1]
                    ]) @ scene.upright.copy()
                    
                    grill_color, grill_color_logp = sample_wheel_color()
                    color_logp += grill_color_logp
                    instance = scene.add_instance(
                        '2412.dat', grill_color, grill_transform)
                    
                    (doodads,
                     doodad_color_logp,
                     doodad_shape_logp) = add_small_doodads(
                        scene, [bumper_brick.snaps[0], bumper_brick.snaps[3]])
                    color_logp += doodad_color_logp
                    shape_logp += doodad_shape_logp
                
                elif grills == 2:
                    grill_color, grill_color_logp = sample_wheel_color()
                    color_logp += grill_color_logp
                    
                    grill_transform = numpy.array([
                        [1, 0, 0, 20],
                        [0, 0, 1, (y+1)*8+6],
                        [0,-1, 0, z1*20 -8],
                        [0, 0, 0, 1]
                    ]) @ scene.upright.copy()
                    instance = scene.add_instance(
                        '2412.dat', grill_color, grill_transform)
                    
                    grill_transform = numpy.array([
                        [1, 0, 0, -20],
                        [0, 0, 1, (y+1)*8+6],
                        [0,-1, 0, z1*20 -8],
                        [0, 0, 0, 1]
                    ]) @ scene.upright.copy()
                    instance = scene.add_instance(
                        '2412.dat', grill_color, grill_transform)
                    
        brick_fill.brick_fill(
            scene,
            x1, x2, y, y+bumper_height, z1+1, z2,
            body_color,
        )
        
        y = y + bumper_height
    
    lower_height, lower_height_logp = choice_logp(
        (1,3,6,9), (0.1, 0.6, 0.2, 0.1))
    shape_logp += lower_height_logp
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
    
    body_length = z2-z1
    hood_setback_options = list(range(body_length-3))
    hood_setback, hood_setback_logp = choice_logp(hood_setback_options)
    shape_logp += hood_setback_logp
    z1 += hood_setback
    
    dimensions = x1, x2, z1, z2, y
    
    (shield_instances,
     shield_color_logp,
     shield_shape_logp,
     rear_dimensions,
     roof_dimensions,
     top_dimensions) = sample_windshield(
        scene, (x1, x2, z1, z2, y), color_palette)
    
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
    
    #roof_body_color, roof_body_color_logp = choice_logp(
    #    (True, False), (0.5, 0.5))
    #color_logp += roof_body_color_logp
    #if roof_body_color:
    #    roof_color = body_color
    #else:
    roof_color, roof_color_logp = choice_logp(color_palette)
    color_logp += roof_color_logp
    brick_fill.brick_fill(
        scene,
        roof_x1, roof_x2, y, y+1, roof_z1, roof_z2,
        roof_color,
    )
    
    #dimensions = x1, x2, z1, z2, y
    
    return [], top_dimensions, color_logp, shape_logp

def sample_roof(scene, dimensions, color_palette):
    color_logp = 0
    shape_logp = 0
    
    x1, x2, z1, z2, y = dimensions
    roof_width = x2-x1
    z_space = z2 - z1
    
    #if z_space < 2:
    #    do_helicopter = 0
    #else:
    #    do_helicopter, do_helicopter_logp = choice_logp([0,1], [0.8,0.2])
    #    logp += do_helicopter_logp
    
    if z_space == 0:
        top_ornament_options = ['none']
        top_ornament_p = [1.0]
    elif z_space == 1:
        top_ornament_options = ['none', 'antenna']
        top_ornament_p = [0.8,0.2]
    elif z_space == 2:
        top_ornament_options = ['none', 'helicopter', 'antenna']
        top_ornament_p = [0.6,0.2,0.2]
    else:
        top_ornament_options = ['none', 'helicopter', 'tail_wings', 'antenna']
        top_ornament_p = [0.4,0.2,0.2,0.2]
    
    top_ornament, top_ornament_logp = choice_logp(
        top_ornament_options, top_ornament_p)
    shape_logp += top_ornament_logp
    
    if top_ornament == 'helicopter':
        post_color, post_color_logp = choice_logp(color_palette)
        color_logp += post_color_logp
        post_z, post_z_logp = choice_logp(range(z1, z2-1))
        shape_logp += post_z_logp
        post_transform = numpy.array([
            [1,0,0,0],
            [0,1,0,(y+2)*8],
            [0,0,1,(post_z+1)*20],
            [0,0,0,1],
        ])@scene.upright
        post = scene.add_instance('2460.dat', post_color, post_transform)
        
        rotor_transform = numpy.array([
            [1,0,0,0],
            [0,1,0,(y+5)*8],
            [0,0,1,(post_z+1)*20],
            [0,0,0,1],
        ])@scene.upright
        rotor = scene.add_instance('2479.dat', 0, rotor_transform)
        
        num_blades, num_blades_logp = choice_logp([2,4])
        shape_logp += num_blades_logp
        blade_length, blade_length_logp = choice_logp([4,6,8])
        shape_logp += blade_length
        blade_color, blade_color_logp = choice_logp(color_palette)
        color_logp += blade_color_logp
        blade1 = brick_fill.make_and_place(
            scene,
            -blade_length-0.5, -0.5, y+5, y+6, post_z+0.5, post_z+1.5,
            blade_color,
        )
        blade2 = brick_fill.make_and_place(
            scene,
            0.5, blade_length+0.5, y+5, y+6, post_z+0.5, post_z+1.5,
            blade_color,
        )
        if num_blades == 4:
            blade3 = brick_fill.make_and_place(
                scene,
                -0.5, 0.5, y+5, y+6, post_z - blade_length + 0.5, post_z + 0.5,
                blade_color,
            )
            blade4 = brick_fill.make_and_place(
                scene,
                -0.5, 0.5, y+5, y+6, post_z + 1.5, post_z + blade_length + 1.5,
                blade_color,
            )
    elif top_ornament == 'tail_wings':
        tail_boom_color, tail_boom_color_logp = choice_logp(color_palette)
        color_logp += tail_boom_color_logp
        
        tail_boom_transform = numpy.array([
            [1,0,0,0],
            [0,1,0,(y+7)*8],
            [0,0,1,z2*20],
            [0,0,0,1],
        ]) @ scene.upright
        
        instance = scene.add_instance(
            '3479.dat', tail_boom_color, tail_boom_transform)
        
        wing_style, wing_style_logp = choice_logp(
            ('2450.dat', '4859.dat', '3475.dat'))
        shape_logp += wing_style_logp
        wing_color, wing_color_logp = choice_logp(color_palette)
        color_logp += wing_color_logp
        
        if wing_style == '4859.dat':
            wing_transform = numpy.array([
                [1,0,0,0],
                [0,1,0,(y+8)*8],
                [0,0,1,z2*20],
                [0,0,0,1],
            ]) @ scene.upright
            instance = scene.add_instance(
                wing_style, wing_color, wing_transform)
        
        elif wing_style == '2450.dat':
            a_transform = numpy.array([
                [1,0,0,-30],
                [0,1,0,(y+8)*8],
                [0,0,1,z2*20+10],
                [0,0,0,1],
            ]) @ scene.upright
            instance = scene.add_instance(
                wing_style, wing_color, a_transform)
            
            b_transform = numpy.array([
                [0,0,-1,30],
                [0,1,0,(y+8)*8],
                [1,0,0,z2*20+10],
                [0,0,0,1],
            ]) @ scene.upright
            instance = scene.add_instance(
                wing_style, wing_color, b_transform)
        
        elif wing_style == '3475.dat':
            a_transform = numpy.array([
                [0,0,1,10],
                [0,1,0,(y+8)*8],
                [-1,0,0,z2*20],
                [0,0,0,1],
            ]) @ scene.upright
            instance = scene.add_instance(
                wing_style, wing_color, a_transform)
            
            b_transform = numpy.array([
                [0,0,-1,-10],
                [0,1,0,(y+8)*8],
                [1,0,0,z2*20],
                [0,0,0,1],
            ]) @ scene.upright
            instance = scene.add_instance(
                wing_style, wing_color, b_transform)
    
    if top_ornament == 'antenna':
        
        antenna_z, antenna_z_logp = choice_logp(range(z1+1,z2+1))
        shape_logp += antenna_z_logp
        
        centrify_color, centrify_color_logp = choice_logp(color_palette)
        color_logp += centrify_color_logp
        
        centrify_transform = numpy.array([
            [1,0,0,0],
            [0,1,0,(y+2)*8],
            [0,0,1,antenna_z*20-10],
            [0,0,0,1],
        ]) @ scene.upright
        instance = scene.add_instance(
            '3794.dat', centrify_color, centrify_transform)
        
        do_antenna, do_antenna_logp = choice_logp((0,1),(0.6,0.4))
        shape_logp += do_antenna_logp
        if do_antenna:
            antenna_color, antenna_color_logp = choice_logp(color_palette)
            color_logp += antenna_color_logp
            
            antenna_transform = numpy.array([
                [1,0,0,0],
                [0,1,0,(y+3)*8],
                [0,0,1,antenna_z*20-10],
                [0,0,0,1],
            ]) @ scene.upright
            instance = scene.add_instance(
                '3957.dat', antenna_color, antenna_transform)
        
        else:
            (instance,
             doodad_color_logp,
             doodad_shape_logp) = add_tall_doodads(scene, [instance.snaps[1]])
            color_logp += doodad_color_logp
            shape_logp += doodad_shape_logp
    
    if top_ornament == 'none' or roof_width == 4:
        
        #do_stickouts, stickout_logp = choice_logp((0,1))
        stickout_style, stickout_logp = choice_logp(
            ('none', 'stickout'))
        shape_logp += stickout_logp
        
        if stickout_style == 'stickout':
            try:
                stickout_z, stickout_z_logp = choice_logp(range(z1+1, z2+1))
            except:
                breakpoint()
            shape_logp += stickout_z_logp
            (stickouts,
             stickouts_color_logp,
             stickouts_shape_logp) = add_stickouts(
                scene,
                color_palette,
                right_location=[x2, y+2, stickout_z],
                left_location=[x1, y+2, stickout_z],
            )
            color_logp += stickouts_color_logp
            shape_logp += stickouts_shape_logp
        
        elif stickout_style == 'stickup':
            try:
                stickup_z, stickup_z_logp = choice_logp(range(z1+1, z2+1))
            except:
                breakpoint()
            shape_logp += stickup_z_logp
            (stickups,
             stickups_color_logp,
             stickups_shape_logp) = add_stickups(
                scene,
                color_palette,
                locations=[[x2, y+2, stickup_z], [x1+1, y+2, stickup_z]],
            )
            color_logp += stickups_color_logp
            shape_logp += stickups_shape_logp
    
    return [], dimensions, color_logp, shape_logp

def sample_windshield(scene, dimensions, color_palette):
    color_logp = 0.
    shape_logp = 0.
    
    x1, x2, z1, z2, y = dimensions
    width = x2 - x1
    
    instances = []
    
    if width == 4:
        windshield_shape, windshield_shape_logp = choice_logp(
            ('3823.dat', # slope 2x6
             '2437.dat', # slope 3x4
             '4866.dat', # different slope 3x4
             '4594.dat', # flat 2x6
             '4215.dat', # flat 1x9 open
             '2483.dat', # helicopter hinge open
            ),
            #(0.2,0.2,0.2,0.2,0.2),
            (0.3,0.1,0.1,0.2,0.1,0.2),
        )
        shape_logp += windshield_shape_logp
        windshield_color, windshield_color_logp = sample_windshield_color()
        color_logp += windshield_color_logp
        windshield_transform = scene.upright.copy()
        if windshield_shape == '3823.dat':
            windshield_transform[1,3] = (y+6)*8
            windshield_transform[2,3] = (z1+1.5)*20
            roof_y = y+6
            roof_z1 = z1+1
            rear_z1 = z1+2
            top_z1 = roof_z1
        elif windshield_shape == '4594.dat':
            windshield_transform[1,3] = (y+6)*8
            windshield_transform[2,3] = (z1+1.5)*20
            roof_y = y+6
            roof_z1 = z1
            rear_z1 = z1+2
            top_z1 = roof_z1
        elif windshield_shape == '2437.dat':
            windshield_transform[1,3] = (y+4)*8
            windshield_transform[2,3] = (z1+1.5)*20
            roof_y = y+4
            roof_z1 = z1+1
            rear_z1 = z1+3
            top_z1 = roof_z1
        elif windshield_shape == '4866.dat':
            windshield_transform[1,3] = (y+4)*8
            windshield_transform[2,3] = (z1+1.5)*20
            roof_y = y+4
            roof_z1 = z1+1
            rear_z1 = z1+3
            top_z1 = roof_z1
        elif windshield_shape == '4215.dat':
            windshield_transform[0,0] *= -1
            windshield_transform[2,2] *= -1
            windshield_transform[1,3] = (y+9)*8
            windshield_transform[2,3] = (z1+0.5)*20
            roof_y = y+9
            roof_z1 = z1
            rear_z1 = z1+3
            top_z1 = roof_z1
        elif windshield_shape == '2483.dat':
            windshield_transform[1,3] = (y+12)*8 + 4
            windshield_transform[2,3] = (z1+3)*20 - 4
            roof_y = y + 12
            roof_z1 = z1+4
            rear_z1 = z1+3
            top_z1 = z1+3
            
            clip_transform = scene.upright.copy()
            clip_transform[1,3] = (y+13)*8
            clip_transform[2,3] = (z1+3)*20 + 10
            clip_color, clip_color_logp = choice_logp(color_palette)
            color_logp += clip_color_logp
            clip = scene.add_instance('4315.dat', clip_color, clip_transform)
        
        i = scene.add_instance(
            windshield_shape, windshield_color, windshield_transform)
        instances.append(i)
    
    elif width == 2:
        windshield_shape, windshield_shape_logp = choice_logp(
            #('2466.dat', '4864.dat'), (0.,1.)
            ('4864.dat',)
        )
        shape_logp += windshield_shape_logp
        windshield_color, windshield_color_logp = sample_windshield_color()
        color_logp += windshield_color_logp
        windshield_transform = scene.upright.copy()
        if windshield_shape == '2466.dat':
            windshield_transform[1,3] = (y+17)*8
            windshield_transform[2,3] = (z1+0.5)*20
            roof_y = y + 17
            roof_z1 = z1
            rear_z1 = z1+3
            top_z1 = roof_z1
        
        if windshield_shape == '4864.dat':
            windshield_transform[0,0] *= -1
            windshield_transform[2,2] *= -1
            windshield_transform[1,3] = (y+6)*8
            windshield_transform[2,3] = (z1+0.5)*20
            roof_y = y + 6
            roof_z1 = z1
            rear_z1 = z1+3
            top_z1 = roof_z1
        
        i = scene.add_instance(
            windshield_shape, windshield_color, windshield_transform)
        instances.append(i)
    
    rear_dimensions = (x1,x2,rear_z1,z2,y)
    roof_dimensions = (x1,x2,roof_z1,z2,roof_y)
    top_dimensions = (x1,x2,top_z1,z2,roof_y)
    return (
        instances,
        color_logp,
        shape_logp,
        rear_dimensions,
        roof_dimensions,
        top_dimensions,
    )

if __name__ == '__main__':
    f = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    
    if not os.path.exists(f):
        os.makedirs(f)
    
    iterate = tqdm.tqdm(range(start, end))
    all_color_logps = []
    all_shape_logps = []
    for i in iterate:
        scene, color_logp, shape_logp = sample_vehicle()
        all_color_logps.append(color_logp)
        all_shape_logps.append(shape_logp)
        #c = sum(all_color_logps)/len(all_color_logps)
        #s = sum(all_shape_logps)/len(all_shape_logps)
        #iterate.set_description('C: %.02f, S: %.02f'%(c,s))
        
        rotate = numpy.array([
            [ 0, 0,-1, 0],
            [ 0, 1, 0,-8],
            [ 1, 0, 0, 0],
            [ 0, 0, 0, 1],
        ])
        for instance in scene.instances.values():
            scene.move_instance(instance, rotate@instance.transform)
        
        scene.export_ldraw(os.path.join(f, 'rcv_%08i.mpd'%i))
    
    print('color:', sum(all_color_logps)/len(all_color_logps))
    print('shape:', sum(all_shape_logps)/len(all_shape_logps))
    
