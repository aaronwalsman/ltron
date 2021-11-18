import os

import numpy

import tqdm

from splendor.obj_mesh import load_mesh

def diff_obj(obj_a, obj_b):
    mesh_a = load_mesh(obj_a)
    mesh_b = load_mesh(obj_b)
    
    va = numpy.array(mesh_a['vertices'])
    vb = numpy.array(mesh_b['vertices'])
    if va.shape != vb.shape:
        return False
    
    vdiff = va - vb
    vmag = numpy.abs(vdiff)
    match = numpy.all(vmag < 0.5)
    
    if match:
        return match
    else:
        print(numpy.max(vmag))
        return match
    
    #with open(obj_a, 'r') as f:
    #    a = f.read()
    #with open(obj_b, 'r') as f:
    #    b = f.read()
    
    #return a == b

def diff_folder(folder_a, folder_b):
    a = set(os.listdir(folder_a))
    b = set(os.listdir(folder_b))
    
    matching = []
    non_matching = []
    missing_a = []
    missing_b = []
    for aa in tqdm.tqdm(a):
        if aa in b:
            a_path = os.path.join(folder_a, aa)
            b_path = os.path.join(folder_b, aa)
            if diff_obj(a_path, b_path):
                matching.append(aa)
            else:
                non_matching.append(aa)
                print('non matching', aa)
        else:
            missing_b.append(aa)
    
    for bb in tqdm.tqdm(b):
        if bb not in a:
            missing_a.append(bb)
    
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    diff_folder(
        '/home/awalsman/.cache/splendor/ltron_assets_low/meshes',
        '/home/awalsman/.cache/splendor/ltron_assets_low/meshes_old',
    )
