import math

import numpy

from ltron.exceptions import LtronException
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.utils import translate_matrix

class LtronInvalidShapeException(LtronException):
    pass

brick_sizes = {
    # plates
    (1, 1, 1): "3024.dat",
    (2, 1, 1): "3023.dat",
    (4, 1, 1): "3710.dat",
    (6, 1, 1): "3666.dat",
    (8, 1, 1): "3460.dat",
    (10,1, 1): "4477.dat",
    (12,1, 1): "60479.dat",

    (2, 1, 2): "3022.dat",
    (3, 1, 2): "3021.dat",
    (4, 1, 2): "3020.dat",
    (6, 1, 2): "3795.dat",
    (8, 1, 2): "3034.dat",
    (10,1, 2): "3832.dat",
    (12,1, 2): "2445.dat",
    (16,1, 2): "4282.dat",

    (4, 1, 4): "3031.dat",
    (6, 1, 4): "3032.dat",
    (8, 1, 4): "3035.dat",
    (12,1, 4): "3029.dat",

    # bricks
    (1, 3, 1): "3005.dat",
    (2, 3, 1): "3004.dat",
    (4, 3, 1): "3010.dat",
    (6, 3, 1): "3009.dat",
    (8, 3, 1): "3008.dat",
    (10,3, 1): "6111.dat",
    (12,3, 1): "6112.dat",

    (2, 3, 2): "3003.dat",
    (3, 3, 2): "3002.dat",
    (4, 3, 2): "3001.dat",
    (6, 3, 2): "2456.dat",
    (8, 3, 2): "3007.dat",
    (10,3, 2): "3006.dat",
    (16,3, 2): "4282.dat",
}

def make_and_place(scene, x1, x2, y1, y2, z1, z2, color):
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if z2 < z1:
        z1, z2 = z2, z1
    w = round(x2-x1)
    l = round(z2-z1)
    h = round(y2-y1)
    translate = translate_matrix(
        (((x2-x1)*0.5 + x1)*20, (y1 + h)*8, ((z2-z1)*0.5 + z1)*20))
    if (w,h,l) in brick_sizes:
        brick_shape = brick_sizes[w,h,l]
        instance = scene.add_instance(
            brick_shape, color, translate@scene.upright)
    elif (l,h,w) in brick_sizes:
        brick_shape = brick_sizes[l,h,w]
        rotate = numpy.array([
            [ 0, 0, 1, 0],
            [ 0, 1, 0, 0],
            [-1, 0, 0, 0],
            [ 0, 0, 0, 1]
        ])
        instance = scene.add_instance(
            brick_shape, color, translate@rotate@scene.upright)
    else:
        raise LtronInvalidShapeException('%s,%s,%s'%(w,h,l))
    
    return instance

def brick_fill(
    scene,
    x1, x2, y1, y2, z1, z2,
    color,
    mask=None,
    use_w=True,
    use_l=True,
    start='min_xz',
    size_selection='largest',
    instances = None,
):
    if mask is None:
        mask = numpy.ones((x2-x1,z2-z1), dtype=bool)
    else:
        mask = numpy.copy(mask)
    
    if instances is None:
        instances = []
    else:
        update_mask(scene, mask, x1, y1*0.25+y2*0.75, z1, instances)
    
    h = y2 - y1
    available_sizes = []
    assert use_w or use_l
    if use_w:
        available_sizes.extend(
            [(ww,hh,ll) for (ww,hh,ll) in brick_sizes if hh == h])
    if use_l:
        available_sizes.extend(
            [(ll,hh,ww) for (ww,hh,ll) in brick_sizes if hh == h])
    
    while numpy.any(mask):
        xs, zs = numpy.where(mask)
        if start == 'random':
            i = numpy.random.randint(xs.shape[0])
            x = xs[i]
            z = zs[i]
        elif start == 'min_xz':
            i = numpy.argmin(xs)
            x = xs[i]
            jj = numpy.where(xs == x)
            zs = zs[jj]
            z = numpy.min(zs)
        elif start == 'min_zx':
            j = numpy.argmin(zs)
            z = zs[j]
            ii = numpy.where(zs == z)
            xs = xs[ii]
            x = numpy.min(xs)
        
        if size_selection == 'random':
            while True:
                i = numpy.random.choice(list(range(len(available_sizes))))
                ww,hh,ll = available_sizes[i]
                area = ww*ll
                if mask[x:x+ww,z:z+ll].sum() == area:
                    break
        
        elif size_selection == 'largest':
            best_size = None
            best_area = 0.
            for ww,hh,ll in available_sizes:
                area = ww*ll
                if area > best_area and mask[x:x+ww,z:z+ll].sum() == area:
                    best_size = ww,hh,ll
                    best_area = area
            ww,hh,ll = best_size
        
        elif size_selection == 'widest':
            best_size = None
            best_width = 0.
            for ww,hh,ll in available_sizes:
                area = ww*ll
                if ww > best_width and mask[x:x+ww,z:z+ll].sum() == area:
                    best_size = ww,hh,ll
                    best_width = ww
            ww,hh,ll = best_size
        
        elif size_selectino == 'longest':
            best_size = None
            best_length = 0.
            for ww,hh,ll in available_sizes:
                area = ww*ll
                if ll > best_length and mask[x:x+ww,z:z+ll].sum() == area:
                    best_size = ww,hh,ll
                    best_length = ll
            ww,hh,ll = best_size
        
        instance = make_and_place(
            scene, x+x1, x+x1+ww, y1, y2, z+z1, z+z1+ll, color)
        update_mask(scene, mask, x1, y1*0.25+y2*0.75, z1, [instance])
        instances.append(instance)
    
    return instances

def update_mask(scene, mask, x_origin, y_origin, z_origin, instances=None):
    if instances is None:
        instances = scene.instances
    for instance in instances:
        (x1,y1,z1), (x2,y2,z2) = scene.get_bbox([instance])
        x1 = max(0, math.floor(x1 / 20. - x_origin + 0.01))
        y1 = y1 / 8.
        z1 = max(0, math.floor(z1 / 20. - z_origin + 0.01))
        x2 = math.ceil(x2 / 20. - x_origin - 0.01)
        y2 = y2 / 8.
        z2 = math.ceil(z2 / 20. - z_origin - 0.01)
        
        w, l = mask.shape
        
        if y1 > y_origin or y2 < y_origin:
            continue
        
        if x1 > w or x2 < 0:
            continue
        
        if z1 > l or z2 < 0:
            continue
        
        mask[x1:x2,z1:z2] = 0

def flip(scene, instances, axis, pivot):
    for instance in instances:
        transform = instance.transform
        x = (transform[axis,3] - pivot*20) * -1 + pivot*20
        transform[axis,3] = x
        scene.move_instance(instance, transform)

def mirror(scene, instances, axis, pivot):
    flip_instances = scene.duplicate_instances(instances)
    flip(scene, flip_instances, axis, pivot)
    return flip_instances

if __name__ == '__main__':
    scene = BrickScene()
    
    instances = brick_fill(
        scene,
        -4, 5, 3, 4, 0, 15,
        4,
        numpy.ones((9,15), dtype=bool),
        use_w=True,
        use_l=True,
        start='min_xz',
        size_selection='largest',
    )
    
    instances2 = scene.duplicate_instances(instances)
    
    flip(scene, instances2, 0, -4)
    
    scene.export_ldraw('./tmp2.mpd')
