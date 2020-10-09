import random
import numpy

# brick shape (w, h)

def create_empty_occupancy(w, d, h):
    return numpy.zeros((w, d, h), bool)

def brick_fits(location, shape, occupancy):
    w, d, h = occupancy.shape
    x, z, y, o = location
    if o == 0:
        brick_w, brick_d = shape
    else:
        brick_d, brick_w = shape
    if x < 0 or x >= w - brick_w + 1:
        return False
    if z < 0 or z >= d - brick_d + 1:
        return False
    if y < 0 or y >= h:
        return False
    
    return numpy.sum(occupancy[x:x+w, z:z+d, y]) == 0

def free_brick_locations(brick_shape, occupancy):
    #return [(0,0,0,0)]
    w, d, h = occupancy.shape
    bw, bd = brick_shape
    
    free_locations = []
    for hh in range(h):
        for dd in range(d-bd+1):
            for ww in range(w-bw+1):
                #if numpy.sum(occupancy[ww:ww+bw, dd:dd+bd, hh]) == 0:
                if brick_fits((ww, dd, hh, 0), brick_shape, occupancy):
                    free_locations.append((ww, dd, hh, 0))
    
    if bw != bd:
        # flip
        for hh in range(h):
            for dd in range(d-bw+1):
                for ww in range(w-bd+1):
                    #if numpy.sum(occupancy[ww:ww+bd, dd:dd+bw, hh]) == 0:
                    if brick_fits((ww, dd, hh, 1), brick_shape, occupancy):
                        free_locations.append((ww, dd, hh, 1))
    
    return free_locations

def free_attachment_locations(bricks, new_brick_shape, occupancy):
    w, d, h = occupancy.shape
    new_brick_w, new_brick_d = new_brick_shape
    free_locations = []
    if not len(bricks):
        return free_brick_locations(new_brick_shape, occupancy[:,:,[0]])
    
    #return [(0,0,1,1)]
    
    for brick in bricks:
        brick_shape, brick_location = brick
        brick_x, brick_z, brick_y, brick_o = brick_location
        if brick_o == 0:
            brick_w, brick_d = brick_shape
        else:
            brick_d, brick_w = brick_shape
        for dd in range(-new_brick_d+1, brick_d):
            z = dd + brick_z
            if z < 0:
                continue
            if z >= d:
                break
            
            for ww in range(-new_brick_w+1, brick_w):
                x = ww + brick_x
                if x < 0:
                    continue
                if z >= d:
                    break
                
                if brick_fits((x,z,brick_y+1,0), new_brick_shape, occupancy):
                    free_locations.append((x,z,brick_y+1,0))
                if brick_fits((x,z,brick_y-1,0), new_brick_shape, occupancy):
                    free_locations.append((x,z,brick_y-1,0))
        
        if new_brick_w != new_brick_d:
            for dd in range(-new_brick_w+1, brick_d):
                z = dd + brick_z
                if z < 0:
                    continue
                if z >= d:
                    break
                
                for ww in range(-new_brick_d+1, brick_w):
                    x = ww + brick_x
                    if x < 0:
                        continue
                    if z >= d:
                        break
                    
                    if brick_fits((x,z,brick_y+1,1),new_brick_shape,occupancy):
                        free_locations.append((x,z,brick_y+1,1))
                    if brick_fits((x,z,brick_y-1,1),new_brick_shape,occupancy):
                        free_locations.append((x,z,brick_y-1,1))
    
    return free_locations

def update_occupancy(brick_shape, location, occupancy):
    x, z, y, o = location
    if not o:
        w, d = brick_shape
    else:
        d, w = brick_shape
    
    occupancy[x:x+w, z:z+d, y] = True

def sample_model(
        min_bricks=3,
        max_bricks=8,
        w = 10,
        d = 10,
        h = 10):
    
    brick_shapes = [
        (1,1),
        (2,1),
        (2,2),
        (3,2),
        (4,2),
        (6,2)
    ]
    
    occupancy = create_empty_occupancy(w,d,h)
    
    num_bricks = random.randint(min_bricks, max_bricks)
    
    bricks = []
    for i in range(num_bricks):
        brick_shape = random.choice(brick_shapes)
        locations = free_attachment_locations(bricks, brick_shape, occupancy)
        if not locations:
            continue
        location = random.choice(locations)
        bricks.append((brick_shape, location))
        update_occupancy(brick_shape, location, occupancy)
        print(brick_shape)
        print(location)
        for hh in range(h):
            print('-'*40)
            print(occupancy[:,:,hh].astype(int))
        print('='*40)
    
    return bricks

def compute_brick_transforms(bricks):
    stud_width = 20.
    brick_height = 24.
    
    #y_up = numpy.eye(4)
    #y_up = numpy.array([
    #        [1, 0, 0, 0],
    #        [0,-1, 0, 0],
    #        [0, 0,-1, 0],
    #        [0, 0, 0, 1]])
    
    transforms = []
    for brick_shape, brick_location in bricks:
        x, z, y, o = brick_location
        brick_w, brick_d = brick_shape
        if not o:
            rotation_offset = numpy.eye(4)
        else:
            #rotation_offset = numpy.eye(4)
            rotation_offset = numpy.array([
                    [ 0, 0,-1, 0],
                    [ 0, 1, 0, 0],
                    [ 1, 0, 0, 0],
                    [ 0, 0, 0, 1]])
        
        pivot_offset = numpy.array([
                [1, 0, 0, stud_width*(brick_w-1)/2.],
                [0, 1, 0, 0],
                [0, 0, 1, stud_width*(brick_d-1)/2.],
                [0, 0, 0, 1]])
        
        translation_offset = numpy.eye(4)
        translation_offset[:3,3] = x*stud_width, y*brick_height, z*stud_width
        if o:
            translation_offset[0,3] += (brick_d-1) * stud_width
        
        transforms.append(numpy.linalg.multi_dot((
                translation_offset,
                rotation_offset,
                pivot_offset)))
                #y_up)))

    return transforms


def bricks_to_mpd(bricks):
    transforms = compute_brick_transforms(bricks)
    
    shapes_to_parts = {
            (1,1) : 3005,
            (2,1) : 3004,
            (2,2) : 3003,
            (3,2) : 3002,
            (4,2) : 3001,
            (6,2) : 2456}
    
    mpd = '''0 FILE Main.ldr
0 Main
0 Name: Main.ldr
0 Author: Randomized python script
0 !LICENSE Yes, you can use this

'''
    
    for (shape, location), transform in zip(bricks, transforms):
        mpd += '1 15 %f %f %f %f %f %f %f %f %f %f %f %f %i.dat\n'%(
                transform[0,3],
                transform[1,3],
                transform[2,3],
                transform[0,0],
                transform[0,1],
                transform[0,2],
                transform[1,0],
                transform[1,1],
                transform[1,2],
                transform[2,0],
                transform[2,1],
                transform[2,2],
                shapes_to_parts[shape])
    
    return mpd

def test():
    bricks = sample_model()
    mpd = bricks_to_mpd(bricks)
    with open('test.mpd', 'w') as f:
        f.write(mpd)
